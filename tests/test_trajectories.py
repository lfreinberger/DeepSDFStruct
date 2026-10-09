"""Tests for trajectory datasets and the trajectory terms of latent-field training.

Covers the motion helpers and trajectory output of ``generate_primitive_dataset``,
``read_trajectory_info`` in ``data.py``, the ``TrajectorySmoothnessLambda`` /
``TrajectoryVelocityLambda`` terms of ``training_latent_field.train`` and the
interpolation test in ``trajectory_eval``.

The training data is generated here: each trajectory is an analytic sphere whose
radius grows linearly in ``t``, so ``dsdf/dt = -dr/dt`` is known exactly.
"""

import json
import pathlib

import numpy as np
import pytest
import torch

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.data import read_trajectory_info
from DeepSDFStruct.deep_sdf.generate_primitive_dataset import (
    _axis_angle_matrix,
    _build_scene_from_params,
    _make_primitive,
    _sample_motion,
    _sample_union_surface,
    _sample_scene_params,
    _scene_params_at,
    _sdf_time_derivative,
    generate_primitive_dataset,
)
from DeepSDFStruct.deep_sdf.training_latent_field import (
    _FixedLatents,
    decoded_time_derivative,
    make_latent_fields,
    train,
    trajectory_smoothness,
)
from DeepSDFStruct.deep_sdf.trajectory_eval import (
    LatentFieldModel,
    evaluate_trajectories,
    evaluate_trajectory_interpolation,
)

UNIT_BOUNDS = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
SPECS_SOURCE = pathlib.Path(
    "DeepSDFStruct/trained_models/test_experiment_latent_field/specs.json"
)


@pytest.fixture(autouse=True)
def _float32():
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(saved)


@pytest.fixture
def scale_range():
    saved = getattr(_make_primitive, "scale_range", None)
    _make_primitive.scale_range = (0.2, 0.4)
    yield (0.2, 0.4)
    if saved is None:
        del _make_primitive.scale_range
    else:
        _make_primitive.scale_range = saved


MOTION = {
    "max_translation": 0.3,
    "max_rotation_deg": 30.0,
    "max_log_scale": 0.3,
    "moving_fraction": 1.0,
}


# --------------------------------------------------------------------------
# motion helpers
# --------------------------------------------------------------------------


def test_axis_angle_matrix_is_a_rotation_about_the_axis():
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    R = _axis_angle_matrix(axis, 0.7)

    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R @ axis, axis, atol=1e-12)
    np.testing.assert_allclose(_axis_angle_matrix(axis, 0.0), np.eye(3))


def test_scene_params_interpolate_the_motion(scale_range):
    rng = np.random.default_rng(0)
    params = _sample_scene_params(["sphere", "box"], 4, UNIT_BOUNDS, True, rng)
    motions = _sample_motion(params, MOTION, UNIT_BOUNDS, rng)

    start = _scene_params_at(params, motions, 0.0)
    end = _scene_params_at(params, motions, 1.0)
    for p, m, a, b in zip(params, motions, start, end):
        np.testing.assert_allclose(a["center"], p["center"])
        np.testing.assert_allclose(a["R"], p["R"], atol=1e-12)
        np.testing.assert_allclose(a["scale_vec"], p["scale_vec"])
        np.testing.assert_allclose(b["center"], p["center"] + m["translation"])
        np.testing.assert_allclose(b["R"] @ b["R"].T, np.eye(3), atol=1e-12)


def test_motion_keeps_end_state_in_range(scale_range):
    rng = np.random.default_rng(3)
    params = _sample_scene_params(["box"], 20, UNIT_BOUNDS, False, rng)
    motion = dict(MOTION, max_translation=5.0, max_log_scale=2.0)
    motions = _sample_motion(params, motion, UNIT_BOUNDS, rng)

    for b in _scene_params_at(params, motions, 1.0):
        assert np.all(b["scale_vec"] >= scale_range[0] - 1e-12)
        assert np.all(b["scale_vec"] <= scale_range[1] + 1e-12)
        margin = b["scale_vec"].max()
        assert np.all(b["center"] >= UNIT_BOUNDS[0] + margin - 1e-12)
        assert np.all(b["center"] <= UNIT_BOUNDS[1] - margin + 1e-12)


def test_zero_moving_fraction_is_static(scale_range):
    rng = np.random.default_rng(1)
    params = _sample_scene_params(["sphere"], 3, UNIT_BOUNDS, True, rng)
    motions = _sample_motion(
        params, dict(MOTION, moving_fraction=0.0), UNIT_BOUNDS, rng
    )
    for p, b in zip(params, _scene_params_at(params, motions, 1.0)):
        np.testing.assert_allclose(b["center"], p["center"])
        np.testing.assert_allclose(b["scale_vec"], p["scale_vec"])


def test_sdf_time_derivative_of_a_translating_sphere():
    """For a sphere moving with velocity d, dsdf/dt = -n . d."""
    params = [
        {
            "type": "sphere",
            "scale_vec": np.full(3, 0.5),
            "R": np.eye(3),
            "center": np.zeros(3),
        }
    ]
    d = np.array([0.2, -0.1, 0.05])
    motions = [
        {
            "translation": d,
            "axis": np.array([0.0, 0.0, 1.0]),
            "angle": 0.0,
            "log_scale": np.zeros(3),
        }
    ]
    points = torch.tensor([[0.7, 0.1, 0.0], [-0.2, 0.3, 0.4], [0.0, 0.0, -0.9]])
    t = 0.5
    got = _sdf_time_derivative(params, motions, t, points, 1e-3).reshape(-1)

    rel = points.double().numpy() - t * d
    n = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    expected = -(n @ d)
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-3)


def test_build_scene_from_params_without_mesh(scale_range):
    params = _sample_scene_params(
        ["cylinder"], 2, UNIT_BOUNDS, True, np.random.default_rng(0)
    )
    sdf, mesh = _build_scene_from_params(params, with_mesh=False)
    assert mesh is None
    assert sdf(torch.zeros(1, 3)).shape == (1, 1)


# --------------------------------------------------------------------------
# dataset generation
# --------------------------------------------------------------------------


def _gen_cfg(root, **overrides):
    cfg = {
        "data_source": str(root),
        "dataset_name": "traj",
        "class_name": "shapes",
        "split_name": "traj.json",
        "num_scenes": 2,
        "primitives_per_scene": 3,
        "primitive_types": ["sphere", "box", "cylinder"],
        "bounds": UNIT_BOUNDS.tolist(),
        "n_uniform": 500,
        "n_surface_per_std": 500,
        "stds": [0.01],
        "scale_range": [0.2, 0.4],
        "random_rotation": True,
        "frames_per_trajectory": 1,
        "motion": MOTION,
        "seed": 7,
        "save_vtp": False,
        "vtp_subdir": "paraview",
        "overwrite": True,
        "instance_start_index": 0,
    }
    cfg.update(overrides)
    return cfg


def test_generate_trajectory_dataset(tmp_path):
    cfg = _gen_cfg(tmp_path, frames_per_trajectory=4)
    summary = generate_primitive_dataset(cfg)
    assert summary["frames_per_trajectory"] == 4

    split = json.loads((tmp_path / "splits" / "traj.json").read_text())
    names = split["traj"]["shapes"]
    assert names == [f"traj_{i}_f{k:03d}" for i in range(2) for k in range(4)]

    sample_dir = tmp_path / ws.sdf_samples_subdir / "traj" / "shapes"
    npz = np.load(sample_dir / "traj_1_f002.npz")
    assert npz["pos"].shape[1] == 5 and npz["neg"].shape[1] == 5
    assert np.all(npz["pos"][:, 3] >= 0) and np.all(npz["neg"][:, 3] < 0)
    assert int(npz["trajectory_id"]) == 1
    assert int(npz["frame_index"]) == 2
    assert float(npz["t"]) == pytest.approx(2 / 3)
    assert np.all(np.isfinite(npz["pos"][:, 4]))

    npyfiles = [f"traj/shapes/{n}.npz" for n in names]
    traj = read_trajectory_info(str(tmp_path), npyfiles)
    assert sorted(traj) == [0, 1]
    assert [sid for sid, _ in traj[1]] == [4, 5, 6, 7]
    assert [t for _, t in traj[0]] == pytest.approx([0, 1 / 3, 2 / 3, 1])


def test_first_frame_equals_the_static_scene(tmp_path):
    static_root = tmp_path / "static"
    traj_root = tmp_path / "traj"
    generate_primitive_dataset(_gen_cfg(static_root, num_scenes=1))
    generate_primitive_dataset(
        _gen_cfg(traj_root, num_scenes=1, frames_per_trajectory=3)
    )

    static = np.load(static_root / ws.sdf_samples_subdir / "traj/shapes/traj_0.npz")
    frame0 = np.load(traj_root / ws.sdf_samples_subdir / "traj/shapes/traj_0_f000.npz")
    assert static["pos"].shape[1] == 4
    assert "trajectory_id" not in static
    np.testing.assert_array_equal(frame0["pos"][:, :4], static["pos"])
    np.testing.assert_array_equal(frame0["neg"][:, :4], static["neg"])


def test_read_trajectory_info_ignores_plain_instances(tmp_path):
    generate_primitive_dataset(_gen_cfg(tmp_path))
    assert read_trajectory_info(str(tmp_path), ["traj/shapes/traj_0.npz"]) == {}


# --------------------------------------------------------------------------
# training terms
# --------------------------------------------------------------------------


def _fields(n, latent_dim=2):
    return make_latent_fields(
        num_scenes=n,
        latent_dim=latent_dim,
        tiling=[1, 1, 1],
        bounds=torch.tensor(UNIT_BOUNDS, dtype=torch.float32),
        device="cpu",
    )


def test_trajectory_smoothness_vanishes_on_linear_paths():
    fields = _fields(4)
    a = torch.randn_like(fields[0].torch_spline.control_points)
    b = torch.randn_like(a)
    for k in range(4):
        fields[k].set_param(a + k / 3 * b)
    frames = [(k, k / 3) for k in range(4)]
    assert trajectory_smoothness(fields, [frames]).item() == pytest.approx(
        0.0, abs=1e-10
    )

    fields[2].set_param(a)  # kink the path
    assert trajectory_smoothness(fields, [frames]).item() > 0
    # two frames carry no curvature information
    assert trajectory_smoothness(fields, [frames[:2]]) is None


def test_decoded_time_derivative_is_the_directional_derivative():
    from DeepSDFStruct.deep_sdf.models import DeepSDFModel
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from DeepSDFStruct.SDF import SDFfromDeepSDF

    specs = json.loads(SPECS_SOURCE.read_text())
    torch.manual_seed(0)
    decoder = ws.init_decoder(specs, "cpu", False)
    model = DeepSDFModel(decoder, torch.zeros(1, 2), device="cpu")
    latents = _FixedLatents()
    probe = LatticeSDFStruct(
        tiling=[1, 1, 1],
        microtile=SDFfromDeepSDF(model),
        parametrization=latents,
        bounds=torch.tensor(UNIT_BOUNDS, dtype=torch.float32),
    )

    x = torch.rand(16, 3) * 1.6 - 0.8
    z = torch.randn(16, 2) * 0.5
    z_dot = torch.randn(16, 2)
    got = decoded_time_derivative(probe, latents, z, z_dot, x)

    # independent reference: the per-point latent gradient (the decoder is
    # pointwise) dotted with z_dot
    z_ref = z.clone().requires_grad_(True)
    latents.z = z_ref
    grad_z = torch.autograd.grad(probe(x).sum(), z_ref)[0]
    torch.testing.assert_close(got, (grad_z * z_dot).sum(-1), atol=1e-6, rtol=1e-5)
    assert got.requires_grad


@pytest.fixture(scope="module")
def trajectory_data(tmp_path_factory):
    """Two trajectories of three frames: spheres with radius r(t) = r0 + 0.2 t."""
    root = tmp_path_factory.mktemp("trajectory_data")
    samples_dir = root / ws.sdf_samples_subdir / "synthetic" / "spheres"
    samples_dir.mkdir(parents=True)
    (root / "splits").mkdir()

    rng = np.random.default_rng(0)
    names = []
    for traj, r0 in enumerate((0.5, 0.6)):
        for k in range(3):
            t = k / 2
            pool = rng.uniform(-1.0, 1.0, size=(8000, 3)).astype(np.float32)
            distance = (np.linalg.norm(pool, axis=1) - (r0 + 0.2 * t)).astype(
                np.float32
            )
            dsdf_dt = np.full_like(distance, -0.2)
            rows = np.concatenate([pool, distance[:, None], dsdf_dt[:, None]], 1)
            name = f"{traj}_f{k}"
            names.append(name)
            np.savez(
                samples_dir / f"{name}.npz",
                pos=rows[distance > 0][:1000],
                neg=rows[distance <= 0][:1000],
                trajectory_id=traj,
                frame_index=k,
                t=t,
            )
    json.dump(
        {"synthetic": {"spheres": names}}, (root / "splits" / "train.json").open("w")
    )
    return root


def _experiment(tmp_path, data_source, **overrides):
    specs = json.loads(SPECS_SOURCE.read_text())
    specs.update({"DataSource": str(data_source)}, **overrides)
    experiment = tmp_path / "experiment"
    experiment.mkdir(exist_ok=True)
    (experiment / "specs.json").write_text(json.dumps(specs, indent=2))
    return experiment


def test_training_with_trajectory_terms(tmp_path, trajectory_data):
    experiment = _experiment(
        tmp_path,
        trajectory_data,
        TrajectorySmoothnessLambda=1.0,
        TrajectoryVelocityLambda=0.1,
        TrajectoryVelocityPoints=8,
    )
    summary = train(experiment, data_source=str(trajectory_data), device="cpu")
    assert np.isfinite(summary["loss"])

    result = evaluate_trajectory_interpolation(
        experiment, data_source=str(trajectory_data), device="cpu", n_samples=256
    )
    # one intermediate frame per trajectory
    assert [(r["trajectory"], r["frame"]) for r in result["frames"]] == [(0, 1), (1, 1)]
    for value in result["mean"].values():
        assert np.isfinite(value)


def test_heldout_evaluation_reconstructs_every_frame(tmp_path, trajectory_data):
    experiment = _experiment(tmp_path, trajectory_data)
    train(experiment, data_source=str(trajectory_data), device="cpu")

    result = evaluate_trajectories(
        experiment,
        split="splits/train.json",
        reconstruct=True,
        data_source=str(trajectory_data),
        device="cpu",
        n_samples=256,
        n_fit=512,
        reconstruct_iters=5,
        velocity_band=0.1,
    )
    assert len(result["control_points"]) == 6  # 2 trajectories x 3 frames
    for key in ("fit_l1", "interp_l1", "vel_ls_rel", "vel_fd_rel"):
        assert np.isfinite(result["mean"][key]), key
    assert 0.0 <= result["mean"]["vel_ls_rel"] <= 1.0 + 1e-6


def test_velocity_residual_is_zero_for_an_expressible_motion(tmp_path, trajectory_data):
    """A target velocity made from the Jacobian itself is fitted exactly."""
    experiment = _experiment(tmp_path, trajectory_data)
    train(experiment, data_source=str(trajectory_data), device="cpu")
    m = LatentFieldModel(experiment, device="cpu")

    torch.manual_seed(0)
    cp = torch.randn_like(m.fields(1)[0].torch_spline.control_points) * 0.1
    cp_dot = torch.randn_like(cp)
    xyz = torch.rand(400, 3) * 1.8 - 0.9
    _, g = m.latent_gradient(cp, xyz)
    field = m.fields(1)[0]
    field.set_param(cp_dot)
    with torch.no_grad():
        v = (g * field(xyz)).sum(-1)

    exact, _, _ = m.velocity_residual(cp, None, (xyz, v), cp_dot=cp_dot)
    assert exact == pytest.approx(0.0, abs=1e-5)
    fitted, fit_rel, _ = m.velocity_residual(
        cp, (xyz[:200], v[:200]), (xyz[:200], v[:200]), ridge=0.0
    )
    assert fitted == pytest.approx(fit_rel)
    assert fitted < 0.05


def test_trajectory_data_trains_without_the_terms(tmp_path, trajectory_data):
    """The extra dsdf/dt column must not break plain training."""
    experiment = _experiment(tmp_path, trajectory_data)
    summary = train(experiment, data_source=str(trajectory_data), device="cpu")
    assert np.isfinite(summary["loss"])


def test_trajectory_terms_need_a_trajectory_dataset(tmp_path):
    root = tmp_path / "plain"
    samples_dir = root / ws.sdf_samples_subdir / "synthetic" / "spheres"
    samples_dir.mkdir(parents=True)
    (root / "splits").mkdir()
    rows = np.random.default_rng(0).uniform(-1, 1, (64, 4)).astype(np.float32)
    for i in range(2):
        np.savez(samples_dir / f"{i}.npz", pos=rows, neg=rows)
    json.dump(
        {"synthetic": {"spheres": ["0", "1"]}},
        (root / "splits" / "train.json").open("w"),
    )
    experiment = _experiment(tmp_path, root, TrajectorySmoothnessLambda=1.0)
    with pytest.raises(ValueError, match="need a trajectory dataset"):
        train(experiment, data_source=str(root), device="cpu")


def test_union_surface_sampling_avoids_buried_surfaces(scale_range):
    """Two overlapping boxes: mesh sampling puts samples deep inside, union not."""
    params = [
        {"type": "box", "scale_vec": np.full(3, 0.4), "R": np.eye(3), "center": c}
        for c in (np.array([-0.15, 0.0, 0.0]), np.array([0.15, 0.0, 0.0]))
    ]
    sdf, mesh = _build_scene_from_params(params)
    np.random.seed(0)
    torch.manual_seed(0)
    union = _sample_union_surface(sdf, mesh, 2000, [0.0])
    assert union.distances.abs().max().item() < 1e-5  # std 0: on the surface

    from DeepSDFStruct.sampling import sample_mesh_surface

    plain = sample_mesh_surface(sdf, mesh, 2000, [0.0])
    assert (plain.distances < -0.05).float().mean().item() > 0.1


def test_union_surface_sampling_projects_curved_primitives():
    """Facets of the sphere mesh lie inside the sphere; samples end up on it."""
    params = [
        {
            "type": "sphere",
            "scale_vec": np.full(3, 0.5),
            "R": np.eye(3),
            "center": np.zeros(3),
        }
    ]
    sdf, mesh = _build_scene_from_params(params)
    out = _sample_union_surface(sdf, mesh, 1000, [0.0])
    torch.testing.assert_close(
        out.samples.norm(dim=-1), torch.full((1000,), 0.5), atol=1e-5, rtol=0
    )
