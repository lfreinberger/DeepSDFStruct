#!/usr/bin/env python3
"""
Primitive-Scene Training Data Generator
=======================================

Generates training scenes of simple geometric primitives (spheres, boxes,
cylinders) randomly positioned, oriented, and scaled inside a bounding box. The
ground-truth SDF is computed *analytically* by combining the primitive SDFs with
the minimum operation (``UnionSDF``).

For each scene the SDF is sampled in two complementary ways:
  - uniformly in the volume (``random_sample_sdf``), and
  - near the surface via Gaussian perturbations at several standard deviations
    (``sample_mesh_surface``).

Trajectories
------------
With ``frames_per_trajectory > 1`` every scene becomes a *trajectory*: the
primitives move (translate, rotate about their centre, rescale) linearly in a
pseudo-time ``t in [0, 1]`` and the scene is sampled at ``K`` equally spaced
frames. Each frame is written as its own instance, so the existing loaders and
trainers see an ordinary dataset; the frame's ``trajectory_id``, ``frame_index``
and ``t`` are stored in the npz so a trainer can group the frames again (see
``read_trajectory_info`` in ``data.py``). Every sample row gets a fifth column,
the time derivative ``dsdf/dt`` of the ground-truth SDF at that point, by a
central finite difference in ``t``. By the level-set equation
``dsdf/dt = -V_n |grad sdf|``, i.e. it encodes the normal velocity of the
boundary -- the quantity a shape sensitivity is made of. Frame 0 of a
trajectory is the static scene the same seed produces with
``frames_per_trajectory = 1``.

The output is written in the layout consumed by ``SDFSamples`` in
``training_latent_field.py``::

    <data_source>/
    ├── SdfSamples/<dataset_name>/<class_name>/<instance>.npz   # pos/neg, [x,y,z,sdf(,dsdf_dt)]
    ├── SdfSamples/<dataset_name>/<vtp_subdir>/<instance>.vtp   # ParaView point clouds
    └── splits/<split_name>.json                                # {dataset:{class:[instance,...]}}

Run directly to generate a dataset using the editable ``CONFIG`` dict at the
bottom of this file::

    python -m DeepSDFStruct.deep_sdf.generate_primitive_dataset
"""

import json
import pathlib
import logging
import datetime
from importlib.metadata import version

import numpy as np
import torch
import trimesh

import DeepSDFStruct
from DeepSDFStruct.sdf_primitives import SphereSDF, BoxSDF, CylinderSDF
from DeepSDFStruct.SDF import TransformedSDF, SDFBase
from DeepSDFStruct.sampling import (
    SampledSDF,
    random_sample_sdf,
    sample_mesh_surface,
    save_points_to_vtp,
)

logger = logging.getLogger(DeepSDFStruct.__name__)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed random 3x3 rotation matrix (Shoemake's method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),  # x
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),  # y
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),  # z
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),  # w
        ]
    )
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _place(
    canonical_sdf: SDFBase,
    canonical_mesh: trimesh.Trimesh,
    R: np.ndarray,
    center: np.ndarray,
) -> tuple[SDFBase, trimesh.Trimesh]:
    """Apply the same rigid transform (rotation R, translation center) to both
    an analytical SDF and its matching mesh so they coincide exactly.

    World SDF is ``s0(R^T (x - center))`` (rigid => distance preserving), which
    ``TransformedSDF`` reproduces with ``rotationMatrix=R^T`` and
    ``translation=R^T @ center``. The matching world vertices are
    ``center + R @ v_canonical``.
    """
    Rt = R.T
    sdf = TransformedSDF(
        canonical_sdf,
        rotationMatrix=Rt.tolist(),  # list-of-lists avoids TransformedSDF's `== [0,0,0]` check
        translation=(Rt @ center).tolist(),
        scaleFactor=1.0,
    )

    if canonical_mesh is None:
        return sdf, None
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    mesh = canonical_mesh.copy()
    mesh.apply_transform(T)
    return sdf, mesh


class _AnisoScaledSDF(SDFBase):
    """Wrap a *unit* canonical SDF with an anisotropic (per-axis) scaling.

    The geometry / zero-level set is exact, but because non-uniform scaling does
    not preserve Euclidean distance, the off-surface magnitude is an
    approximation. We multiply by ``min(scale)`` so the field stays a valid
    1-Lipschitz signed distance (it never overestimates the true distance and is
    exact along the least-scaled axis). Boxes do not use this wrapper because
    ``BoxSDF`` represents anisotropic extents exactly.
    """

    def __init__(self, sdf: SDFBase, scale_vec):
        super().__init__()
        self.sdf = sdf
        s = torch.as_tensor(scale_vec, dtype=torch.float32).reshape(1, 3)
        self.register_buffer("scale_vec", s)
        self.correction = float(s.min().item())

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        sv = self.scale_vec.to(device=queries.device, dtype=queries.dtype)
        return self.sdf._compute(queries / sv) * self.correction

    def _get_domain_bounds(self) -> torch.Tensor:
        return self.sdf._get_domain_bounds() * self.scale_vec.reshape(-1)


def _make_primitive(
    prim_type: str, rng: np.random.Generator
) -> tuple[SDFBase, trimesh.Trimesh, np.ndarray]:
    """Build a primitive SDF and a matching trimesh with an independent random
    per-axis scale (``scale_vec`` = semi-size along x/y/z).

    Boxes use exact per-axis ``BoxSDF`` extents. Spheres/cylinders start from a
    unit canonical shape and receive an anisotropic scale wrapper (ellipsoid /
    elliptical cylinder); the returned SDF is the object to feed into ``_place``
    (rotation + translation), and the mesh is the matching pre-scaled mesh.
    """
    s_lo, s_hi = _make_primitive.scale_range
    scale_vec = rng.uniform(s_lo, s_hi, size=3)  # independent x/y/z semi-sizes
    sdf, mesh = _primitive_sdf_and_mesh(prim_type, scale_vec)
    return sdf, mesh, scale_vec


def _primitive_sdf_and_mesh(
    prim_type: str, scale_vec: np.ndarray, with_mesh: bool = True
) -> tuple[SDFBase, trimesh.Trimesh | None]:
    """Canonical (unplaced) primitive SDF and matching mesh for ``scale_vec``.

    ``with_mesh=False`` skips the mesh (returns ``None``) when only the SDF is
    needed, e.g. for the finite-difference time derivative.
    """
    mesh = None
    if prim_type == "sphere":
        unit_sdf = SphereSDF(center=[0.0, 0.0, 0.0], radius=1.0)
        sdf = _AnisoScaledSDF(unit_sdf, scale_vec)
        if with_mesh:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            mesh.apply_scale(scale_vec.tolist())  # -> ellipsoid, semi-axes scale_vec
    elif prim_type == "box":
        extents = (2.0 * scale_vec).tolist()  # exact anisotropic box
        sdf = BoxSDF(center=[0.0, 0.0, 0.0], extents=extents)
        if with_mesh:
            mesh = trimesh.creation.box(extents=extents)
    elif prim_type == "cylinder":
        # unit cylinder: radius 1 (x/y), height 2 (half-height 1 along z), which
        # matches trimesh.creation.cylinder(radius=1, height=2) below.
        unit_sdf = CylinderSDF(
            point_a=[0.0, 0.0, -1.0], point_b=[0.0, 0.0, 1.0], radius=1.0
        )
        sdf = _AnisoScaledSDF(unit_sdf, scale_vec)
        if with_mesh:
            mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
            mesh.apply_scale(scale_vec.tolist())  # elliptical section + scaled height
    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")

    return sdf, mesh


def _sample_scene_params(
    primitive_types: list[str],
    n_primitives: int,
    bounds: np.ndarray,
    random_rotation: bool,
    rng: np.random.Generator,
) -> list[dict]:
    """Draw the parameters of a scene: per primitive its type, per-axis
    semi-sizes ``scale_vec``, rotation ``R`` and ``center``."""
    bounds_lo, bounds_hi = bounds[0], bounds[1]
    s_lo, s_hi = _make_primitive.scale_range
    params: list[dict] = []

    for _ in range(n_primitives):
        prim_type = str(rng.choice(primitive_types))
        scale_vec = rng.uniform(s_lo, s_hi, size=3)  # independent x/y/z semi-sizes

        # keep the primitive (roughly) inside the box; out-of-bounds samples are
        # rejected later regardless, this just avoids wasting samples.
        margin = float(np.max(scale_vec))
        lo = np.minimum(bounds_lo + margin, bounds_hi - margin)
        hi = np.maximum(bounds_lo + margin, bounds_hi - margin)
        center = rng.uniform(lo, hi)

        R = _random_rotation_matrix(rng) if random_rotation else np.eye(3)
        params.append(
            {"type": prim_type, "scale_vec": scale_vec, "R": R, "center": center}
        )
    return params


def _build_scene_from_params(
    params: list[dict], with_mesh: bool = True
) -> tuple[SDFBase, trimesh.Trimesh | None]:
    """Compose a scene SDF (union of placed primitives) and the concatenated
    surface mesh used for near-surface sampling (``None`` if ``with_mesh`` is
    off)."""
    sdfs: list[SDFBase] = []
    meshes: list[trimesh.Trimesh] = []
    for p in params:
        canonical_sdf, canonical_mesh = _primitive_sdf_and_mesh(
            p["type"], p["scale_vec"], with_mesh=with_mesh
        )
        sdf, mesh = _place(canonical_sdf, canonical_mesh, p["R"], p["center"])
        sdfs.append(sdf)
        meshes.append(mesh)

    scene_sdf = sdfs[0]
    for s in sdfs[1:]:
        scene_sdf = scene_sdf + s  # UnionSDF via torch.minimum

    scene_mesh = trimesh.util.concatenate(meshes) if with_mesh else None
    return scene_sdf, scene_mesh


def _build_scene(
    primitive_types: list[str],
    n_primitives: int,
    bounds: np.ndarray,
    random_rotation: bool,
    rng: np.random.Generator,
) -> tuple[SDFBase, trimesh.Trimesh]:
    """Compose a random scene SDF (union of placed primitives) and the
    concatenated surface mesh used for near-surface sampling."""
    params = _sample_scene_params(
        primitive_types, n_primitives, bounds, random_rotation, rng
    )
    return _build_scene_from_params(params)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about the unit vector ``axis`` by ``angle`` (Rodrigues)."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _sample_motion(
    params: list[dict], motion: dict, bounds: np.ndarray, rng: np.random.Generator
) -> list[dict]:
    """Draw a linear motion over ``t in [0, 1]`` for every primitive.

    A primitive moves with probability ``motion["moving_fraction"]``; a moving
    one translates by up to ``max_translation`` (random direction), rotates
    about its centre by up to ``max_rotation_deg`` (random axis) and rescales
    each semi-axis by a factor in ``exp(+-max_log_scale)``. The end centre is
    kept inside the box with the larger of the start / end margins, and the
    end scale inside ``scale_range``.
    """
    s_lo, s_hi = _make_primitive.scale_range
    max_translation = float(motion.get("max_translation", 0.0))
    max_rotation = np.deg2rad(float(motion.get("max_rotation_deg", 0.0)))
    max_log_scale = float(motion.get("max_log_scale", 0.0))
    moving_fraction = float(motion.get("moving_fraction", 1.0))

    motions: list[dict] = []
    for p in params:
        static = {
            "translation": np.zeros(3),
            "axis": np.array([0.0, 0.0, 1.0]),
            "angle": 0.0,
            "log_scale": np.zeros(3),
        }
        if rng.random() >= moving_fraction:
            motions.append(static)
            continue

        translation = _random_unit_vector(rng) * rng.uniform(0.0, max_translation)
        axis = _random_unit_vector(rng)
        angle = rng.uniform(-max_rotation, max_rotation)
        log_scale = rng.uniform(-max_log_scale, max_log_scale, size=3)

        scale_end = np.clip(p["scale_vec"] * np.exp(log_scale), s_lo, s_hi)
        log_scale = np.log(scale_end / p["scale_vec"])

        margin = float(max(np.max(p["scale_vec"]), np.max(scale_end)))
        lo = np.minimum(bounds[0] + margin, bounds[1] - margin)
        hi = np.maximum(bounds[0] + margin, bounds[1] - margin)
        translation = np.clip(p["center"] + translation, lo, hi) - p["center"]

        motions.append(
            {
                "translation": translation,
                "axis": axis,
                "angle": angle,
                "log_scale": log_scale,
            }
        )
    return motions


def _scene_params_at(params: list[dict], motions: list[dict], t: float) -> list[dict]:
    """Scene parameters at pseudo-time ``t`` (linear in translation, rotation
    angle and log-scale; ``t`` outside ``[0, 1]`` extrapolates)."""
    out = []
    for p, m in zip(params, motions):
        out.append(
            {
                "type": p["type"],
                "scale_vec": p["scale_vec"] * np.exp(t * m["log_scale"]),
                "R": _axis_angle_matrix(m["axis"], t * m["angle"]) @ p["R"],
                "center": p["center"] + t * m["translation"],
            }
        )
    return out


def _sdf_time_derivative(
    params: list[dict], motions: list[dict], t: float, points: torch.Tensor, dt: float
) -> torch.Tensor:
    """``d sdf / dt`` at ``points`` by a central finite difference in ``t``.

    Exact up to O(dt^2) where the scene SDF is smooth in ``t``; at the medial
    sets of the union (``torch.minimum``) and inside non-exact anisotropic
    fields it is the one-sided mix the min selects.
    """
    sdf_plus, _ = _build_scene_from_params(
        _scene_params_at(params, motions, t + dt), with_mesh=False
    )
    sdf_minus, _ = _build_scene_from_params(
        _scene_params_at(params, motions, t - dt), with_mesh=False
    )
    with torch.no_grad():
        return (sdf_plus(points) - sdf_minus(points)) / (2.0 * dt)


def _sample_union_surface(
    scene_sdf: SDFBase,
    scene_mesh: trimesh.Trimesh,
    n_samples: int,
    stds: list[float],
    buried_tol: float = 0.02,
    projection_steps: int = 3,
    max_rounds: int = 50,
) -> SampledSDF:
    """Near-surface samples around the *visible* surface of the union.

    ``scene_mesh`` concatenates every primitive's surface, so with overlapping
    primitives a large share of it lies buried inside other primitives;
    ``sample_mesh_surface`` then puts "near-surface" samples deep inside the
    solid. Here surface points with ``sdf < -buried_tol`` are rejected
    (``buried_tol`` exceeds the facet error of the curved primitives' meshes),
    the rest is projected onto the analytic zero level set by Newton steps
    ``p <- p - sdf(p) n(p)`` and perturbed along ``n = grad sdf / |grad sdf|``.
    Every std gets its own ``n_samples`` surface points.
    """
    samples = []
    for std in stds:
        kept, count = [], 0
        for _ in range(max_rounds):
            pts, _ = trimesh.sample.sample_surface(scene_mesh, 2 * n_samples)
            p = torch.tensor(pts, dtype=torch.float32)
            with torch.no_grad():
                p = p[scene_sdf(p).reshape(-1) > -buried_tol]
            kept.append(p)
            count += len(p)
            if count >= n_samples:
                break
        p = torch.cat(kept)[:n_samples]
        with torch.enable_grad():
            for _ in range(projection_steps + 1):
                p = p.detach().requires_grad_(True)
                d = scene_sdf(p).reshape(-1, 1)
                g = torch.autograd.grad(d.sum(), p)[0]
                normal = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                if _ < projection_steps:
                    p = p - d * normal
        p, normal = p.detach(), normal.detach()
        t = torch.randn(len(p), 1) * std
        samples.append(p + t * normal)
    queries = torch.vstack(samples)
    with torch.no_grad():
        distances = scene_sdf(queries)
    return SampledSDF(samples=queries, distances=distances)


def _filter_to_bounds(sampled: SampledSDF, bounds: np.ndarray) -> SampledSDF:
    """Drop any sample whose coordinates fall outside ``bounds`` (e.g. when
    near-surface Gaussian perturbations push points beyond the box)."""
    lo = torch.tensor(bounds[0], dtype=sampled.samples.dtype)
    hi = torch.tensor(bounds[1], dtype=sampled.samples.dtype)
    inside = ((sampled.samples >= lo) & (sampled.samples <= hi)).all(dim=1)
    return SampledSDF(
        samples=sampled.samples[inside], distances=sampled.distances[inside]
    )


def scene_parameters(cfg: dict, idx: int) -> tuple[list[dict], list[dict] | None]:
    """Primitive parameters and motions (``None`` for a static dataset) of
    scene / trajectory ``idx`` exactly as ``generate_primitive_dataset`` draws
    them for ``cfg``. Rebuild frame ``t`` with
    ``_build_scene_from_params(_scene_params_at(params, motions, t))``, e.g.
    for the analytic ground truth of a trajectory dataset.
    """
    _make_primitive.scale_range = tuple(cfg["scale_range"])
    bounds = np.asarray(cfg["bounds"], dtype=np.float64)
    rng = np.random.default_rng(int(cfg["seed"]) + idx)
    params = _sample_scene_params(
        primitive_types=list(cfg["primitive_types"]),
        n_primitives=int(cfg["primitives_per_scene"]),
        bounds=bounds,
        random_rotation=bool(cfg["random_rotation"]),
        rng=rng,
    )
    motions = None
    if int(cfg.get("frames_per_trajectory", 1)) > 1:
        motions = _sample_motion(params, dict(cfg.get("motion", {})), bounds, rng)
    return params, motions


def generate_primitive_dataset(cfg: dict) -> dict:
    """Generate a primitive-scene SDF dataset according to ``cfg``.

    Returns the dataset summary dict (also written to ``summary.json``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    data_source = pathlib.Path(cfg["data_source"])
    dataset_name = cfg["dataset_name"]
    class_name = cfg["class_name"]
    bounds = np.asarray(cfg["bounds"], dtype=np.float64)

    # propagate the size range to the primitive factory
    _make_primitive.scale_range = tuple(cfg["scale_range"])

    sample_dir = data_source / "SdfSamples" / dataset_name / class_name
    vtp_dir = data_source / "SdfSamples" / dataset_name / cfg["vtp_subdir"]
    split_path = data_source / "splits" / cfg["split_name"]

    sample_dir.mkdir(parents=True, exist_ok=True)
    if cfg["save_vtp"]:
        vtp_dir.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(cfg.get("frames_per_trajectory", 1))
    if n_frames < 1:
        raise ValueError(f"frames_per_trajectory must be >= 1, got {n_frames}")
    motion = dict(cfg.get("motion", {}))
    dsdf_dt_step = float(cfg.get("dsdf_dt_step", 1e-3))

    instance_names: list[str] = []
    n_scenes = int(cfg["num_scenes"])
    for i in range(n_scenes):
        idx = int(cfg["instance_start_index"]) + i
        if n_frames == 1:
            frame_names = [f"{dataset_name}_{idx}"]
        else:
            frame_names = [f"{dataset_name}_{idx}_f{k:03d}" for k in range(n_frames)]
        instance_names.extend(frame_names)

        npz_paths = [sample_dir / f"{name}.npz" for name in frame_names]
        if all(path.is_file() for path in npz_paths) and not cfg["overwrite"]:
            logger.info(f"[skip] {npz_paths[0]} ... ({len(npz_paths)} exist)")
            continue

        # deterministic, per-scene seeding (covers numpy + torch + trimesh.sample)
        scene_seed = int(cfg["seed"]) + idx
        np.random.seed(scene_seed % (2**32 - 1))
        torch.manual_seed(scene_seed)
        params, motions = scene_parameters(cfg, idx)

        for k, (name, npz_path) in enumerate(zip(frame_names, npz_paths)):
            t = k / (n_frames - 1) if n_frames > 1 else 0.0
            frame_params = (
                params if motions is None else _scene_params_at(params, motions, t)
            )

            with torch.no_grad():
                scene_sdf, scene_mesh = _build_scene_from_params(frame_params)

                uniform = random_sample_sdf(
                    scene_sdf,
                    bounds=bounds.tolist(),
                    n_samples=int(cfg["n_uniform"]),
                    sampling_strategy="uniform",
                )
                if cfg.get("surface_sampling", "mesh") == "union":
                    surface = _sample_union_surface(
                        scene_sdf,
                        scene_mesh,
                        int(cfg["n_surface_per_std"]),
                        list(cfg["stds"]),
                    )
                else:
                    surface = sample_mesh_surface(
                        scene_sdf,
                        scene_mesh,
                        int(cfg["n_surface_per_std"]),
                        list(cfg["stds"]),
                    )
                combined = uniform + surface
                # near-surface Gaussian perturbations can push points past the
                # box; reject anything outside the bounds so no sample lies outside.
                combined = _filter_to_bounds(combined, bounds)

                rows = combined.stacked
                if motions is not None:
                    dsdf_dt = _sdf_time_derivative(
                        params, motions, t, combined.samples, dsdf_dt_step
                    )
                    rows = torch.hstack((rows, dsdf_dt.reshape(-1, 1)))

            rows = rows.detach().cpu().numpy()
            is_pos = rows[:, 3] >= 0.0
            extra = {}
            if motions is not None:
                extra = {"trajectory_id": idx, "frame_index": k, "t": t}
            np.savez(npz_path, neg=rows[~is_pos], pos=rows[is_pos], **extra)

            if cfg["save_vtp"]:
                save_points_to_vtp(vtp_dir / f"{name}.vtp", combined.stacked)

            logger.info(
                f"[{i + 1}/{n_scenes}] {name}: "
                f"{int(is_pos.sum())} pos / {int((~is_pos).sum())} neg -> {npz_path}"
            )

    # split json: {dataset: {class: [instance, ...]}}
    split = {dataset_name: {class_name: instance_names}}
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=4)
    logger.info(f"Wrote split {split_path}")

    summary = {
        "dataset_name": dataset_name,
        "class_name": class_name,
        "num_scenes": n_scenes,
        "primitives_per_scene": int(cfg["primitives_per_scene"]),
        "primitive_types": list(cfg["primitive_types"]),
        "bounds": bounds.tolist(),
        "n_uniform": int(cfg["n_uniform"]),
        "n_surface_per_std": int(cfg["n_surface_per_std"]),
        "stds": list(cfg["stds"]),
        "scale_range": list(cfg["scale_range"]),
        "random_rotation": bool(cfg["random_rotation"]),
        "surface_sampling": cfg.get("surface_sampling", "mesh"),
        "frames_per_trajectory": n_frames,
        "motion": motion,
        "dsdf_dt_step": dsdf_dt_step,
        "seed": int(cfg["seed"]),
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sdf_struct_version": version("DeepSDFStruct"),
    }
    summary_path = data_source / "SdfSamples" / dataset_name / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Wrote summary {summary_path}")

    return summary


CONFIG = {
    "data_source": "/Users/lukas/projects/projectsPhd/TrainingData",  # DataSource root
    "dataset_name": "2026_06_25_primitive_shapes_4",
    "class_name": "shapes",
    "split_name": "primitives_train_4.json",  # written under <data_source>/splits/
    "num_scenes": 4,  # paper N=100
    "primitives_per_scene": 10,  # paper 10
    "primitive_types": ["sphere", "box", "cylinder"],
    "bounds": [[-1, -1, -1], [1, 1, 1]],  # Omega_box
    "n_uniform": 100_000,  # uniform samples / scene
    "n_surface_per_std": 500_000,  # x len(stds) => 1,000,000 near-surface
    "stds": [0.005, 0.0001],  # paper sigma1, sigma2
    "scale_range": [0.1, 0.5],  # characteristic half-size of primitives
    "random_rotation": True,
    # "mesh": perturb points of every primitive's surface (incl. surfaces buried in
    # other primitives); "union": only the visible surface of the union, projected
    # onto the analytic zero level set (_sample_union_surface)
    "surface_sampling": "mesh",
    # > 1: every scene is a trajectory of this many frames (see module docstring)
    "frames_per_trajectory": 1,
    "motion": {
        "max_translation": 0.3,  # per primitive, over t in [0, 1]
        "max_rotation_deg": 30.0,  # about the primitive's centre
        "max_log_scale": 0.3,  # per-axis scale factor in exp(+-0.3)
        "moving_fraction": 0.5,  # share of primitives that move
    },
    "dsdf_dt_step": 1e-3,  # finite-difference step in t for dsdf/dt
    "seed": 42,
    "save_vtp": True,  # ParaView point clouds
    "vtp_subdir": "paraview",
    "overwrite": True,
    "instance_start_index": 10000,
}


if __name__ == "__main__":
    generate_primitive_dataset(CONFIG)
