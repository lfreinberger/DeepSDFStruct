"""
Trajectory interpolation test for spline-latent-field models
============================================================

Checks whether *linear interpolation in latent space follows the motion* of a
trajectory dataset (``generate_primitive_dataset`` with
``frames_per_trajectory > 1``), which is what a gradient step of a shape
optimizer relies on.

For every trajectory, only the control points of the first and the last frame
are used: the intermediate frames are decoded from their linear interpolation
(at the frame's ``t``) and compared with that frame's ground-truth samples.
As reference the same frames are decoded from their own trained control points
(the fit). A model whose latent space encodes motion has an interpolation
error close to its fit error; one that blends shapes (fade-out / fade-in)
does not.

Metrics per frame, on the frame's samples:
  - ``l1``: clamped L1 (``ClampingDistance`` of the experiment), the training
    loss,
  - ``sign_acc``: share of samples whose inside/outside sign is right.

Run::

    python -m DeepSDFStruct.deep_sdf.trajectory_eval <experiment_dir> \\
        [--checkpoint latest.pth] [--samples 20000]
"""

import argparse
import logging
import os
import pathlib
import json

import numpy as np
import torch

import DeepSDFStruct
import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.data import (
    get_instance_filenames,
    read_trajectory_info,
    unpack_sdf_samples,
)
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.deep_sdf.training_latent_field import (
    get_spec_with_default,
    load_latent_fields,
    make_latent_fields,
)
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.SDF import SDFfromDeepSDF

logger = logging.getLogger(DeepSDFStruct.__name__)


def _frame_metrics(pred, sdf_gt, clamp_dist):
    l1 = (
        (pred.clamp(-clamp_dist, clamp_dist) - sdf_gt.clamp(-clamp_dist, clamp_dist))
        .abs()
        .mean()
    )
    sign_acc = ((pred >= 0) == (sdf_gt >= 0)).float().mean()
    return float(l1.item()), float(sign_acc.item())


def evaluate_trajectory_interpolation(
    experiment_directory,
    checkpoint: str = "latest.pth",
    data_source: str | None = None,
    device: str | None = None,
    n_samples: int | None = 20000,
    seed: int = 0,
) -> dict:
    """Fit vs. linear-interpolation error on the intermediate frames of every
    trajectory in the experiment's training split.

    Returns ``{"frames": [...], "mean": {...}}``; each frame record holds
    ``trajectory``, ``frame``, ``t`` and ``fit_l1``, ``interp_l1``,
    ``fit_sign_acc``, ``interp_sign_acc``. ``n_samples`` (balanced pos / neg)
    are drawn per frame; ``None`` uses all of them.
    """
    experiment_directory = str(experiment_directory)
    specs = ws.load_experiment_specifications(experiment_directory)
    if data_source is None:
        data_source = specs["DataSource"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    ckpt_noext = checkpoint[:-4] if checkpoint.endswith(".pth") else checkpoint
    ckpt_name = ckpt_noext + ".pth"
    decoder = ws.load_trained_model(experiment_directory, ckpt_noext, device=device)
    decoder.eval()
    geom_dimension = decoder.geom_dimension

    latent_size = specs["CodeLength"]
    latent_dim = (
        int(torch.tensor(latent_size).sum().item())
        if isinstance(latent_size, list)
        else int(latent_size)
    )
    tiling = get_spec_with_default(specs, "Tiling", [1, 1, 1])
    bounds = torch.tensor(
        get_spec_with_default(
            specs, "BoundsParamSpace", [[-1.0] * geom_dimension, [1.0] * geom_dimension]
        ),
        dtype=torch.float32,
        device=device,
    )
    spline_degrees = get_spec_with_default(specs, "SplineDegrees", None)
    clamp_dist = float(get_spec_with_default(specs, "ClampingDistance", 0.1))

    with open(pathlib.Path(data_source) / specs["TrainSplit"], "r") as f:
        npyfiles = get_instance_filenames(data_source, json.load(f))
    trajectories = read_trajectory_info(data_source, npyfiles)
    if not trajectories:
        raise ValueError("the training split holds no trajectory frames")

    common = dict(
        latent_dim=latent_dim,
        tiling=tiling,
        bounds=bounds,
        device=device,
        degrees=spline_degrees,
    )
    latent_fields = make_latent_fields(num_scenes=len(npyfiles), **common)
    load_latent_fields(experiment_directory, ckpt_name, latent_fields, device=device)
    # holds the interpolated control points
    interp_field = make_latent_fields(num_scenes=1, **common)[0]

    dummy_latents = torch.zeros((1, latent_dim), device=device, dtype=torch.float32)
    model = DeepSDFModel(decoder, dummy_latents, device=device)

    def decode(field, xyz):
        struct = LatticeSDFStruct(
            tiling=tiling,
            microtile=SDFfromDeepSDF(model),
            parametrization=field,
            bounds=bounds,
        )
        return struct(xyz)

    records = []
    with torch.no_grad():
        for traj, frames in sorted(trajectories.items()):
            if len(frames) < 3:
                continue
            (sid_0, t_0), (sid_1, t_1) = frames[0], frames[-1]
            cp_0 = latent_fields[sid_0].torch_spline.control_points
            cp_1 = latent_fields[sid_1].torch_spline.control_points
            for position, (sid, t) in enumerate(frames[1:-1], start=1):
                filename = os.path.join(
                    data_source, ws.sdf_samples_subdir, npyfiles[sid]
                )
                samples = unpack_sdf_samples(filename, geom_dimension, n_samples).to(
                    device
                )
                xyz = samples[:, :geom_dimension]
                sdf_gt = samples[:, geom_dimension : geom_dimension + 1]

                alpha = (t - t_0) / (t_1 - t_0)
                interp_field.set_param((1.0 - alpha) * cp_0 + alpha * cp_1)

                fit_l1, fit_sign = _frame_metrics(
                    decode(latent_fields[sid], xyz), sdf_gt, clamp_dist
                )
                int_l1, int_sign = _frame_metrics(
                    decode(interp_field, xyz), sdf_gt, clamp_dist
                )
                records.append(
                    {
                        "trajectory": traj,
                        "frame": position,
                        "t": t,
                        "fit_l1": fit_l1,
                        "interp_l1": int_l1,
                        "fit_sign_acc": fit_sign,
                        "interp_sign_acc": int_sign,
                    }
                )

    keys = ("fit_l1", "interp_l1", "fit_sign_acc", "interp_sign_acc")
    mean = {
        k: float(np.mean([r[k] for r in records])) if records else float("nan")
        for k in keys
    }
    return {"frames": records, "mean": mean}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("experiment_directory")
    parser.add_argument("--checkpoint", default="latest.pth")
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--samples", type=int, default=20000, help="samples per frame (0: all)"
    )
    parser.add_argument("--out", default=None, help="write the result as json")
    args = parser.parse_args(argv)

    result = evaluate_trajectory_interpolation(
        args.experiment_directory,
        checkpoint=args.checkpoint,
        data_source=args.data_source,
        device=args.device,
        n_samples=args.samples or None,
    )
    for r in result["frames"]:
        print(
            f"traj {r['trajectory']} frame {r['frame']} (t={r['t']:.3f}): "
            f"L1 fit {r['fit_l1']:.4f} / interp {r['interp_l1']:.4f}, "
            f"sign fit {r['fit_sign_acc']:.3f} / interp {r['interp_sign_acc']:.3f}"
        )
    m = result["mean"]
    print(
        f"mean: L1 fit {m['fit_l1']:.4f} / interp {m['interp_l1']:.4f}, "
        f"sign fit {m['fit_sign_acc']:.3f} / interp {m['interp_sign_acc']:.3f}"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    main()
