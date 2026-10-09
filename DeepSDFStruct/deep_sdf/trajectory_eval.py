"""
Trajectory tests for spline-latent-field models
===============================================

Checks whether the latent space of a model *encodes motion*, which is what a
gradient step of a shape optimizer relies on. Needs a trajectory dataset
(``generate_primitive_dataset`` with ``frames_per_trajectory > 1``), whose
samples carry the ground-truth time derivative ``dsdf/dt``.

Every frame of a trajectory gets control points: on the training split the
trained ones, on a held-out split a reconstruction with the frozen decoder
(``reconstruct=True``; each frame independently, from zero). Then, per
intermediate frame:

  - ``fit_l1`` / ``fit_sign_acc``: the frame decoded from its own control
    points (clamped L1 at ``ClampingDistance``; share of samples with the
    right inside/outside sign),
  - ``interp_l1`` / ``interp_sign_acc``: the frame decoded from the linear
    interpolation of the first and last frame's control points. Close to the
    fit when a straight latent path follows the motion; worse when the model
    blends shapes (fade-out / fade-in),
  - ``vel_ls_rel``: how well the decoder can express the true boundary
    velocity at all, ``min |J c_dot - dsdf/dt| / |dsdf/dt|`` over control-point
    velocities ``c_dot``, with ``J`` the Jacobian of the decoded SDF with
    respect to the control points. Fitted on one set of near-surface points,
    measured on another; 0 means every motion of the data is a direction of
    the latent space, 1 means none is,
  - ``vel_fd_rel``: the same residual for the finite-difference velocity of
    the neighbouring frames' control points, i.e. whether those (trained or
    independently reconstructed) control points move consistently in time.

Run::

    python -m DeepSDFStruct.deep_sdf.trajectory_eval <experiment_dir> \\
        [--split splits/<test>.json --reconstruct] [--checkpoint latest.pth]
"""

import argparse
import json
import logging
import os
import pathlib

import numpy as np
import torch

import DeepSDFStruct
import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.data import (
    _read_pos_neg,
    get_instance_filenames,
    read_trajectory_info,
)
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.deep_sdf.training_latent_field import (
    _FixedLatents,
    get_spec_with_default,
    load_latent_fields,
    make_latent_fields,
)
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.SDF import SDFfromDeepSDF

logger = logging.getLogger(DeepSDFStruct.__name__)


class LatentFieldModel:
    """Frozen decoder and lattice settings of a latent-field experiment."""

    def __init__(self, experiment_directory, checkpoint="latest.pth", device=None):
        self.experiment_directory = str(experiment_directory)
        self.specs = ws.load_experiment_specifications(self.experiment_directory)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_noext = checkpoint[:-4] if checkpoint.endswith(".pth") else checkpoint
        self.checkpoint = ckpt_noext + ".pth"
        self.decoder = ws.load_trained_model(
            self.experiment_directory, ckpt_noext, device=self.device
        )
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.geom_dimension = self.decoder.geom_dimension

        specs = self.specs
        latent_size = specs["CodeLength"]
        self.latent_dim = (
            int(torch.tensor(latent_size).sum().item())
            if isinstance(latent_size, list)
            else int(latent_size)
        )
        self.tiling = get_spec_with_default(specs, "Tiling", [1, 1, 1])
        dim = self.geom_dimension
        self.bounds = torch.tensor(
            get_spec_with_default(
                specs, "BoundsParamSpace", [[-1.0] * dim, [1.0] * dim]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.degrees = get_spec_with_default(specs, "SplineDegrees", None)
        self.clamp_dist = float(get_spec_with_default(specs, "ClampingDistance", 0.1))
        self.code_bound = get_spec_with_default(specs, "CodeBound", None)
        self.code_reg = float(
            get_spec_with_default(specs, "CodeRegularizationLambda", 0.0)
        )

        dummy = torch.zeros((1, self.latent_dim), device=self.device)
        self.model = DeepSDFModel(self.decoder, dummy, device=self.device)
        self.probe_latents = _FixedLatents()
        self.probe = self.struct(self.probe_latents)

    def fields(self, n):
        """``n`` latent fields with the experiment's spline topology."""
        return make_latent_fields(
            num_scenes=n,
            latent_dim=self.latent_dim,
            tiling=self.tiling,
            bounds=self.bounds,
            device=self.device,
            degrees=self.degrees,
        )

    def trained_fields(self, num_scenes):
        fields = self.fields(num_scenes)
        load_latent_fields(
            self.experiment_directory, self.checkpoint, fields, device=self.device
        )
        return fields

    def struct(self, parametrization):
        return LatticeSDFStruct(
            tiling=self.tiling,
            microtile=SDFfromDeepSDF(self.model),
            parametrization=parametrization,
            bounds=self.bounds,
        )

    def decode(self, control_points, xyz, field=None):
        """SDF at ``xyz`` for ``control_points`` (``field`` is reused if given)."""
        field = field if field is not None else self.fields(1)[0]
        field.set_param(control_points)
        return self.struct(field)(xyz)

    def latent_gradient(self, control_points, xyz):
        """``(z, df/dz)`` at ``xyz``; the decoder is pointwise, so the gradient
        of the summed output is each point's own gradient."""
        field = self.fields(1)[0]
        field.set_param(control_points)
        with torch.no_grad():
            z = field(xyz)
        z = z.clone().requires_grad_(True)
        self.probe_latents.z = z
        try:
            f = self.probe(xyz)
        finally:
            self.probe_latents.z = None
        return z.detach(), torch.autograd.grad(f.sum(), z)[0]

    def reconstruct(self, xyz, sdf, iters=800, lr=5e-3, batch=8192, seed=0):
        """Control points fitted to the samples with the frozen decoder (Adam,
        clamped L1, from zero; ``lr`` drops by 5x for the last 30 %)."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        field = self.fields(1)[0]
        cp = field.torch_spline.control_points
        with torch.no_grad():
            cp.zero_()
        struct = self.struct(field)
        opt = torch.optim.Adam([cp], lr=lr)
        sdf_c = sdf.clamp(-self.clamp_dist, self.clamp_dist)
        for it in range(iters):
            if it == int(0.7 * iters):
                for g in opt.param_groups:
                    g["lr"] = lr / 5
            idx = torch.randint(len(xyz), (min(batch, len(xyz)),), generator=gen)
            idx = idx.to(xyz.device)
            pred = struct(xyz[idx]).clamp(-self.clamp_dist, self.clamp_dist)
            loss = (pred - sdf_c[idx]).abs().mean()
            if self.code_reg > 0:
                loss = loss + self.code_reg * cp.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if self.code_bound is not None:
                with torch.no_grad():
                    cp.clamp_(-float(self.code_bound), float(self.code_bound))
        return cp.detach().clone()

    def velocity_residual(
        self, control_points, fit, evaluate, cp_dot=None, iters=400, ridge=1e-3
    ):
        """Relative residual ``|J c_dot - v| / |v|`` on the ``evaluate`` points.

        ``fit`` and ``evaluate`` are ``(xyz, v)`` pairs (``v`` = ``dsdf/dt``).
        Without ``cp_dot`` the velocity is fitted on ``fit`` by ridge least
        squares (L-BFGS from zero; ``ridge`` weighs the mean square of
        ``c_dot`` against the relative squared residual). There are more
        control-point velocities than a few ten thousand points can pin down,
        so without the ridge and enough points the fit interpolates the fit
        points and generalizes badly. With ``cp_dot`` given, ``fit`` is unused.
        Returns ``(residual on evaluate, residual on fit or nan, cp_dot)``.
        """
        xyz_e, v_e = evaluate
        _, g_e = self.latent_gradient(control_points, xyz_e)
        dot_field = self.fields(1)[0]
        dot_cp = dot_field.torch_spline.control_points

        if cp_dot is None:
            xyz_f, v_f = fit
            _, g_f = self.latent_gradient(control_points, xyz_f)
            with torch.no_grad():
                dot_cp.zero_()
            opt = torch.optim.LBFGS(
                [dot_cp],
                lr=1.0,
                max_iter=iters,
                history_size=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                line_search_fn="strong_wolfe",
            )
            scale = v_f.pow(2).mean().clamp_min(1e-12)

            def closure():
                opt.zero_grad(set_to_none=True)
                r = (g_f * dot_field(xyz_f)).sum(-1) - v_f
                loss = r.pow(2).mean() / scale + ridge * dot_cp.pow(2).mean()
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                r_f = (g_f * dot_field(xyz_f)).sum(-1) - v_f
                rel_fit = float((r_f.norm() / v_f.norm().clamp_min(1e-12)).item())
        else:
            dot_field.set_param(cp_dot)
            rel_fit = float("nan")

        with torch.no_grad():
            r = (g_e * dot_field(xyz_e)).sum(-1) - v_e
            rel = r.norm() / v_e.norm().clamp_min(1e-12)
        return float(rel.item()), rel_fit, dot_cp.detach().clone()


def load_frame_samples(filename, n, seed=0, geom_dimension=3):
    """``n`` samples of one frame, half from each sign (as in training), in a
    random order. Rows are ``[x, y, z, sdf(, dsdf_dt)]``."""
    pos, neg = _read_pos_neg(np.load(filename))
    rng = np.random.default_rng(seed)
    half = n // 2
    pos = pos[rng.choice(len(pos), min(half, len(pos)), replace=False)]
    neg = neg[rng.choice(len(neg), min(n - len(pos), len(neg)), replace=False)]
    rows = np.concatenate([pos, neg], 0)
    rows = rows[rng.permutation(len(rows))]
    rows = rows[~np.isnan(rows[:, geom_dimension])]
    return torch.from_numpy(rows).float()


def _frame_metrics(pred, sdf_gt, clamp_dist):
    l1 = (
        (pred.clamp(-clamp_dist, clamp_dist) - sdf_gt.clamp(-clamp_dist, clamp_dist))
        .abs()
        .mean()
    )
    sign_acc = ((pred >= 0) == (sdf_gt >= 0)).float().mean()
    return float(l1.item()), float(sign_acc.item())


def evaluate_trajectories(
    experiment_directory,
    split: str | None = None,
    reconstruct: bool = False,
    checkpoint: str = "latest.pth",
    data_source: str | None = None,
    device: str | None = None,
    n_samples: int | None = 20000,
    n_fit: int = 200_000,
    reconstruct_iters: int = 800,
    reconstruct_lr: float = 5e-3,
    velocity: bool = True,
    velocity_band: float = 0.01,
    velocity_fit_points: int = 60000,
    velocity_eval_points: int = 20000,
    seed: int = 0,
    model: LatentFieldModel | None = None,
    control_points: dict | None = None,
) -> dict:
    """The trajectory tests of the module docstring on ``split`` (default: the
    experiment's ``TrainSplit``, which uses the trained control points).

    ``reconstruct`` fits every frame with the frozen decoder on ``n_fit``
    samples; metrics are always measured on ``n_samples`` other samples
    (``None``: all remaining). Velocity residuals need the ``dsdf/dt`` column
    use up to ``velocity_fit_points`` / ``velocity_eval_points`` disjoint
    points within ``|sdf| < velocity_band``. ``control_points`` (as returned
    before) skips the reconstruction and reuses them.

    Returns ``{"frames": [...], "mean": {...}, "control_points": {...}}``;
    ``control_points`` maps ``(trajectory, frame)`` to the frame's control
    points.
    """
    m = model or LatentFieldModel(experiment_directory, checkpoint, device)
    dim = m.geom_dimension
    if data_source is None:
        data_source = m.specs["DataSource"]
    split = split or m.specs["TrainSplit"]
    with open(pathlib.Path(data_source) / split, "r") as f:
        npyfiles = get_instance_filenames(data_source, json.load(f))
    trajectories = read_trajectory_info(data_source, npyfiles)
    if not trajectories:
        raise ValueError(f"the split {split} holds no trajectory frames")
    trained = None if reconstruct else m.trained_fields(len(npyfiles))

    given = control_points
    records, control_points = [], {}
    for traj, frames in sorted(trajectories.items()):
        if len(frames) < 3:
            continue
        cps, evals, vel_sets = [], [], []
        for position, (sid, t) in enumerate(frames):
            filename = os.path.join(data_source, ws.sdf_samples_subdir, npyfiles[sid])
            n_eval = n_samples or 0
            # the same rows as for the reconstruction, so evaluation rows stay
            # unseen; the velocity band is taken from all of them
            n_load = n_fit + n_eval if n_samples else 10**12
            rows = load_frame_samples(filename, n_load, seed + sid, dim).to(m.device)
            fit_rows = rows[: min(n_fit, len(rows) // 2)] if reconstruct else rows[:0]
            eval_rows = rows[len(fit_rows) :][: n_samples or None]
            if given is not None:
                cp = given[(traj, position)].to(m.device)
            elif reconstruct:
                cp = m.reconstruct(
                    fit_rows[:, :dim],
                    fit_rows[:, dim : dim + 1],
                    iters=reconstruct_iters,
                    lr=reconstruct_lr,
                    seed=seed + sid,
                )
                logger.info(f"reconstructed trajectory {traj} frame {position}")
            else:
                cp = trained[sid].torch_spline.control_points.detach().clone()
            cps.append(cp)
            control_points[(traj, position)] = cp
            evals.append(eval_rows)
            if velocity and rows.shape[1] > dim + 1:
                band = rows[rows[:, dim].abs() < velocity_band]
                n_e = min(velocity_eval_points, len(band) // 2)
                n_f = min(velocity_fit_points, len(band) - n_e)
            if velocity and rows.shape[1] > dim + 1 and n_e >= 8:
                vel_sets.append(
                    (
                        (band[:n_f, :dim], band[:n_f, dim + 1]),
                        (band[-n_e:, :dim], band[-n_e:, dim + 1]),
                    )
                )
            else:
                vel_sets.append(None)

        t_0, t_1 = frames[0][1], frames[-1][1]
        for position in range(1, len(frames) - 1):
            t = frames[position][1]
            rows = evals[position]
            xyz, sdf_gt = rows[:, :dim], rows[:, dim : dim + 1]
            alpha = (t - t_0) / (t_1 - t_0)
            cp_interp = (1.0 - alpha) * cps[0] + alpha * cps[-1]
            with torch.no_grad():
                fit_l1, fit_sign = _frame_metrics(
                    m.decode(cps[position], xyz), sdf_gt, m.clamp_dist
                )
                int_l1, int_sign = _frame_metrics(
                    m.decode(cp_interp, xyz), sdf_gt, m.clamp_dist
                )
            record = {
                "trajectory": traj,
                "frame": position,
                "t": t,
                "fit_l1": fit_l1,
                "interp_l1": int_l1,
                "fit_sign_acc": fit_sign,
                "interp_sign_acc": int_sign,
            }
            if vel_sets[position] is not None:
                fit_set, eval_set = vel_sets[position]
                record["vel_ls_rel"], record["vel_ls_fit_rel"], _ = m.velocity_residual(
                    cps[position], fit_set, eval_set
                )
                dt = frames[position + 1][1] - frames[position - 1][1]
                cp_dot = (cps[position + 1] - cps[position - 1]) / dt
                record["vel_fd_rel"], _, _ = m.velocity_residual(
                    cps[position], None, eval_set, cp_dot=cp_dot
                )
            records.append(record)

    keys = sorted({k for r in records for k in r} - {"trajectory", "frame", "t"})
    mean = {
        k: float(np.mean([r[k] for r in records if k in r])) if records else np.nan
        for k in keys
    }
    return {"frames": records, "mean": mean, "control_points": control_points}


def evaluate_trajectory_interpolation(experiment_directory, **kwargs) -> dict:
    """``evaluate_trajectories`` on the training split with the trained
    control points (no reconstruction)."""
    return evaluate_trajectories(experiment_directory, reconstruct=False, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("experiment_directory")
    parser.add_argument("--split", default=None, help="default: TrainSplit")
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="fit the frames with the frozen decoder (held-out splits)",
    )
    parser.add_argument("--checkpoint", default="latest.pth")
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--samples", type=int, default=20000, help="evaluation samples per frame"
    )
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--out", default=None, help="write the result as json")
    parser.add_argument(
        "--save-control-points", default=None, help="torch.save the control points"
    )
    args = parser.parse_args(argv)

    result = evaluate_trajectories(
        args.experiment_directory,
        split=args.split,
        reconstruct=args.reconstruct,
        checkpoint=args.checkpoint,
        data_source=args.data_source,
        device=args.device,
        n_samples=args.samples or None,
        velocity=not args.no_velocity,
    )
    for r in result["frames"]:
        line = (
            f"traj {r['trajectory']} frame {r['frame']} (t={r['t']:.3f}): "
            f"L1 fit {r['fit_l1']:.4f} / interp {r['interp_l1']:.4f}, "
            f"sign fit {r['fit_sign_acc']:.3f} / interp {r['interp_sign_acc']:.3f}"
        )
        if "vel_ls_rel" in r:
            line += f", vel LS {r['vel_ls_rel']:.3f} / FD {r['vel_fd_rel']:.3f}"
        print(line)
    print("mean: " + ", ".join(f"{k} {v:.4f}" for k, v in result["mean"].items()))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({k: result[k] for k in ("frames", "mean")}, f, indent=2)
    if args.save_control_points:
        torch.save(result["control_points"], args.save_control_points)
    return result


if __name__ == "__main__":
    main()
