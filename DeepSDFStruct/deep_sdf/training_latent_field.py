#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import os
import sys
import json
import math
import time
import signal
import random
import logging
import pathlib
import datetime
import socket

import numpy as np
import torch
import torch.utils.data as data_utils
from importlib.metadata import version

import splinepy

import DeepSDFStruct.deep_sdf
import DeepSDFStruct.deep_sdf.workspace as ws
import DeepSDFStruct.deep_sdf.data

from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.deep_sdf.plotting import plot_logs
from DeepSDFStruct.mesh import create_3D_mesh, export_surface_mesh
from DeepSDFStruct.deep_sdf.data import SDFSamples

import mlflow
import mlflow.pytorch

logger = logging.getLogger(DeepSDFStruct.__name__)


class ClampedL1Loss(torch.nn.Module):
    def __init__(self, clamp_val=0.1):
        super().__init__()
        self.clamp_val = clamp_val
        self.loss = torch.nn.L1Loss()

    def forward(self, input, target):
        input_clamped = input.clamp(-self.clamp_val, self.clamp_val)
        target_clamped = target.clamp(-self.clamp_val, self.clamp_val)
        return self.loss(input_clamped, target_clamped)


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        raise NotImplementedError


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = float(value)

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = float(initial)
        self.interval = int(interval)
        self.factor = float(factor)

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = float(initial)
        self.warmed_up = float(warmed_up)
        self.length = int(length)

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


class CosineAnnealingLRSchedule(LearningRateSchedule):
    def __init__(self, initial, final, total_epochs):
        self.initial = float(initial)
        self.final = float(final)
        self.total_epochs = int(total_epochs)

    def get_learning_rate(self, epoch):
        if epoch >= self.total_epochs:
            return self.final
        return self.final + 0.5 * (self.initial - self.final) * (
            1 + math.cos(math.pi * epoch / self.total_epochs)
        )


def get_spec_with_default(specs, key, default):
    return specs[key] if key in specs else default


def _seed_worker(worker_id):
    """Seed each DataLoader worker deterministically.

    Without this, numpy/random state in workers is not reproducible across runs,
    so the random subsampling in ``unpack_sdf_samples_from_ram`` (which uses the
    ``random`` module) and shuffling are non-deterministic with num_workers > 1.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class _FixedLatents(torch.nn.Module):
    """Parametrization that hands the lattice a precomputed latent per query.

    Lets ``LatticeSDFStruct`` decode latents that are not a spline evaluation,
    e.g. ``z + s * z_dot`` for the time-derivative term. ``z`` must have one
    row per query the struct passes in (all queries inside its bounds).
    """

    def __init__(self):
        super().__init__()
        self.z = None

    def forward(self, queries):
        if self.z is None or self.z.shape[0] != queries.shape[0]:
            raise RuntimeError("_FixedLatents: latents do not match the queries")
        return self.z


def _control_points(latent_field):
    return latent_field.torch_spline.control_points


def trajectory_smoothness(latent_fields, trajectories):
    """Mean squared second difference of the control points along each
    trajectory (frames assumed equally spaced in ``t``).

    Zero when the latent field moves on a straight line at constant speed
    between the frames, i.e. when linear interpolation of the control points
    reproduces the intermediate frames. Trajectories with fewer than three
    frames contribute nothing.
    """
    terms = []
    for frames in trajectories:
        if len(frames) < 3:
            continue
        cps = torch.stack([_control_points(latent_fields[sid]) for sid, _ in frames])
        terms.append((cps[2:] - 2.0 * cps[1:-1] + cps[:-2]).pow(2).mean())
    if not terms:
        return None
    return torch.stack(terms).mean()


def latent_time_derivative(latent_fields, frames, position, queries):
    """``dz/dt`` of the latent field at ``queries`` for frame ``position`` of a
    trajectory: central difference of the neighbouring frames' latent fields
    (one-sided at the ends). ``frames`` is ``[(scene_index, t), ...]``."""
    a = max(position - 1, 0)
    b = min(position + 1, len(frames) - 1)
    (sid_a, t_a), (sid_b, t_b) = frames[a], frames[b]
    return (latent_fields[sid_b](queries) - latent_fields[sid_a](queries)) / (t_b - t_a)


def decoded_time_derivative(probe, probe_latents, z, z_dot, queries):
    """``d f(x, z(x, t)) / dt = (df/dz) . z_dot`` per query point.

    The directional derivative is taken with respect to a per-point scalar
    ``s`` in ``f(x, z + s z_dot)`` at ``s = 0``: the decoder is pointwise, so
    the gradient of the summed output gives each point's own derivative. The
    graph is kept so the result can be trained.
    """
    s = torch.zeros(queries.shape[0], device=queries.device, requires_grad=True)
    probe_latents.z = z + s.unsqueeze(-1) * z_dot
    try:
        pred = probe(queries)
    finally:
        probe_latents.z = None
    return torch.autograd.grad(pred.sum(), s, create_graph=True)[0]


def _make_lr_schedule(sched_spec):
    if isinstance(sched_spec, (int, float)):
        return ConstantLearningRateSchedule(sched_spec)

    t = sched_spec.get("Type", "Constant").lower()
    if t == "constant":
        return ConstantLearningRateSchedule(sched_spec["Value"])
    if t == "step":
        return StepLearningRateSchedule(
            sched_spec["Initial"], sched_spec["Interval"], sched_spec["Factor"]
        )
    if t == "warmup":
        return WarmupLearningRateSchedule(
            sched_spec["Initial"], sched_spec["WarmedUp"], sched_spec["Length"]
        )
    if t == "cosine":
        return CosineAnnealingLRSchedule(
            sched_spec["Initial"], sched_spec["Final"], sched_spec["TotalEpochs"]
        )
    raise ValueError(f"Unknown LR schedule type: {sched_spec}")


def save_model(experiment_directory, filename, decoder, epoch):
    model_params_dir = ws.get_model_params_dir(
        experiment_directory, create_if_nonexistent=True
    )
    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):
    optim_dir = ws.get_optimizer_params_dir(
        experiment_directory, create_if_nonexistent=True
    )
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optim_dir, filename),
    )


def save_latent_fields(
    experiment_directory, filename, latent_fields, epoch, num_scenes, latent_dim, device
):
    lat_dir = ws.get_latent_codes_dir(experiment_directory, create_if_nonexistent=True)

    # Dummy latents only for compatibility with ws.load_latent_vectors / get_model
    dummy_latent_codes = torch.zeros((num_scenes, latent_dim), device="cpu")

    torch.save(
        {
            "epoch": epoch,
            "latent_fields_state_dict": latent_fields.state_dict(),
            "latent_codes": dummy_latent_codes,
        },
        os.path.join(lat_dir, filename),
    )


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )
    plot_logs(
        experiment_directory,
        show_lr=True,
        filename=os.path.join(experiment_directory, ws.logplot_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def load_optimizer(experiment_directory, filename, optimizer):
    optim_dir = ws.get_optimizer_params_dir(
        experiment_directory, create_if_nonexistent=False
    )
    path = os.path.join(optim_dir, filename)
    if not os.path.isfile(path):
        return 0
    data = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return int(data.get("epoch", 0))


def load_latent_fields(experiment_directory, filename, latent_fields, device):
    lat_dir = ws.get_latent_codes_dir(experiment_directory, create_if_nonexistent=False)
    path = os.path.join(lat_dir, filename)
    if not os.path.isfile(path):
        return 0
    data = torch.load(path, map_location=device)
    latent_fields.load_state_dict(data["latent_fields_state_dict"])
    return int(data.get("epoch", 0))


def build_template_spline(latent_dim, tiling, bounds, degrees=None):
    """
    Builds a splinepy.BSpline in [mins,maxs]^d with a control point grid that matches tiling.
    """
    dim = len(tiling)
    if degrees is None:
        degrees = [1] * dim
    if len(degrees) != dim:
        raise ValueError(f"degrees must have length {dim}, got {degrees}")

    # bounds: (2, dim) -> mins/maxs
    # bounds_np = np.asarray(bounds, dtype=float)
    mins = bounds[0].detach().cpu().numpy()
    maxs = bounds[1].detach().cpu().numpy()

    # clamped knot vectors per dim: [min]*(p+1) + [max]*(p+1)
    knot_vectors = []
    for i in range(dim):
        p = int(degrees[i])
        kv = [float(mins[i])] * (p + 1) + [float(maxs[i])] * (p + 1)
        knot_vectors.append(kv)

    # initial CP count = prod(p+1)
    ncp0 = int(np.prod([p + 1 for p in degrees]))
    control_points = [[0.0] * int(latent_dim) for _ in range(ncp0)]

    sp = splinepy.BSpline(degrees, knot_vectors, control_points)

    # insert internal knots in [min,max] like reconstruction script
    for i_dim, n_box in enumerate(tiling):
        n_box = int(n_box)
        if n_box == 1:
            continue
        knots = np.linspace(mins[i_dim], maxs[i_dim], n_box + 1)[1:-1]
        sp.insert_knots(i_dim, knots)

    return sp


def make_latent_fields(
    num_scenes, latent_dim, tiling, bounds, device, init_std=0.01, degrees=None
):
    """
    Create one SplineParametrization (with learnable control points) per scene.
    All splines share identical topology (degrees/knot vectors/control point count).
    """
    template = build_template_spline(latent_dim, tiling, bounds, degrees=degrees)
    deg = list(template.degrees)
    kvs = [list(kv) for kv in template.knot_vectors]
    ncp = int(np.asarray(template.control_points).shape[0])

    latent_fields = torch.nn.ModuleList()
    for _ in range(int(num_scenes)):
        sp = splinepy.BSpline(deg, kvs, [[0.0] * int(latent_dim) for _ in range(ncp)])
        param = SplineParametrization(sp, device=device)
        # init CP params
        for p in param.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=float(init_std))
        latent_fields.append(param)

    return latent_fields


def get_mean_spline_param_magnitude(latent_fields):
    mags = []
    for lf in latent_fields:
        for p in lf.parameters():
            mags.append(p.data.norm().item())
    return float(sum(mags) / len(mags)) if mags else 0.0


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log:
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def train(
    experiment_directory,
    data_source=None,
    continue_from=None,
    device=None,
    use_mlflow: bool = False,
    mlflow_run_name: str | None = None,
    mlflow_tags: dict | None = None,
    mlflow_log_every_n_batches: int = 50,
):
    """
    Train decoder + spline latent fields.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Normalize before first use: the log line below concatenates it, which
    # raises TypeError for a PathLike argument.
    experiment_directory = str(experiment_directory)
    logging.debug("running " + experiment_directory)
    specs = ws.load_experiment_specifications(experiment_directory)
    logging.info("Experiment description: \n" + specs["Description"])
    use_mlflow = bool(use_mlflow) and (mlflow.active_run() is not None)
    if use_mlflow:
        if mlflow_tags:
            mlflow.set_tags(mlflow_tags)

        # Core params from specs.json
        mlflow.log_param("Description", specs.get("Description", ""))
        mlflow.log_param("DataSource", specs.get("DataSource", ""))
        mlflow.log_param("TrainSplit", specs.get("TrainSplit", ""))
        mlflow.log_param("NetworkArch", specs.get("NetworkArch", ""))

        mlflow.log_param("CodeLength", specs.get("CodeLength", None))
        mlflow.log_param("NumEpochs", specs.get("NumEpochs", None))
        mlflow.log_param("SamplesPerScene", specs.get("SamplesPerScene", None))
        mlflow.log_param("ScenesPerBatch", specs.get("ScenesPerBatch", None))
        mlflow.log_param("DataLoaderThreads", specs.get("DataLoaderThreads", None))

        mlflow.log_param("ClampingDistance", specs.get("ClampingDistance", None))
        mlflow.log_param(
            "CodeRegularizationLambda", specs.get("CodeRegularizationLambda", None)
        )
        mlflow.log_param("CodeBound", specs.get("CodeBound", None))
        mlflow.log_param(
            "TrajectorySmoothnessLambda", specs.get("TrajectorySmoothnessLambda", 0.0)
        )
        mlflow.log_param(
            "TrajectoryVelocityLambda", specs.get("TrajectoryVelocityLambda", 0.0)
        )

        mlflow.log_param("Tiling", json.dumps(specs.get("Tiling", None)))
        mlflow.log_param(
            "BoundsParamSpace", json.dumps(specs.get("BoundsParamSpace", None))
        )
        mlflow.log_param("SplineInitStd", specs.get("SplineInitStd", None))
        mlflow.log_param("SplineDegrees", json.dumps(specs.get("SplineDegrees", None)))
        mlflow.log_param(
            "LearningRateSchedule", json.dumps(specs.get("LearningRateSchedule", None))
        )

        # NetworkSpecs (a few highlights)
        ns = specs.get("NetworkSpecs", {})
        mlflow.log_param("geom_dimension", ns.get("geom_dimension", None))
        mlflow.log_param("dims", json.dumps(ns.get("dims", None)))
        mlflow.log_param("use_tanh", ns.get("use_tanh", None))
        mlflow.log_param("weight_norm", ns.get("weight_norm", None))

        # Runtime params
        mlflow.log_param("device", str(device))

        # Log the full specs as artifact (best reproducibility)
        specs_dump_path = os.path.join(experiment_directory, "mlflow_specs_dump.json")
        with open(specs_dump_path, "w") as f:
            json.dump(specs, f, indent=2)
        mlflow.log_artifact(specs_dump_path)

    if data_source is None:
        data_source = specs["DataSource"]

    is_quantum = specs.get("NetworkArch", "") == "quantum_deep_sdf_decoder"

    if device is None:
        # QNN statevector simulation gets no benefit from GPU at small qubit
        # counts — lightning.qubit runs faster on CPU.
        device = (
            "cpu" if is_quantum else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    host_name = socket.gethostname()
    logging.info(f"training on {host_name} with {device}")

    seed = get_spec_with_default(specs, "seed", 42)
    logging.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    # DataParallel is not supported for QNN (lightning.qubit is not GPU-aware).
    data_parallel = (not is_quantum) and torch.cuda.device_count() > 1
    decoder = ws.init_decoder(specs, device, data_parallel)
    geom_dimension = decoder.geom_dimension

    logging.debug(specs["NetworkSpecs"])
    latent_size = specs["CodeLength"]
    if isinstance(latent_size, list):
        latent_dim = int(torch.tensor(latent_size).sum().item())
    else:
        latent_dim = int(latent_size)

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_fields(
            experiment_directory,
            "latest.pth",
            latent_fields,
            epoch,
            num_scenes,
            latent_dim,
            device,
        )

    def save_checkpoints(epoch):
        save_model(experiment_directory, f"{epoch}.pth", decoder, epoch)
        save_optimizer(experiment_directory, f"{epoch}.pth", optimizer_all, epoch)
        save_latent_fields(
            experiment_directory,
            f"{epoch}.pth",
            latent_fields,
            epoch,
            num_scenes,
            latent_dim,
            device,
        )

    num_epochs = int(specs["NumEpochs"])
    num_samp_per_scene = int(specs["SamplesPerScene"])
    scene_per_batch = int(specs["ScenesPerBatch"])
    batch_split = int(get_spec_with_default(specs, "BatchSplit", 1))

    clamp_dist = float(get_spec_with_default(specs, "ClampingDistance", 0.1))
    minT, maxT = -clamp_dist, clamp_dist
    enforce_minmax = bool(get_spec_with_default(specs, "EnforceMinMax", True))

    # bounds/tiling for param-space pipeline (THIS is the key difference vs classic training)
    # bounds are used in LatticeSDFStruct for:
    #   - mapping samples -> [0,1]^d before spline eval
    #   - periodic transform to DeepSDF microtile coords in [-1,1]^d
    tiling = get_spec_with_default(specs, "Tiling", [1, 1, 1])
    if len(tiling) != geom_dimension:
        raise ValueError(
            f"Tiling length must match geom_dimension={geom_dimension}. Got {tiling}"
        )

    bounds_param_space = get_spec_with_default(
        specs, "BoundsParamSpace", [[-1.0] * geom_dimension, [1.0] * geom_dimension]
    )
    bounds_param_space = torch.tensor(
        bounds_param_space, dtype=torch.float32, device=device
    )

    spline_init_std = float(get_spec_with_default(specs, "SplineInitStd", 0.01))
    spline_degrees = get_spec_with_default(specs, "SplineDegrees", None)

    code_reg_lambda = float(
        get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    )
    code_reg_target = get_spec_with_default(
        specs, "CodeRegularizationTarget", "evaluated"
    )  # "control_points", "evaluated", or "both"

    code_bound = get_spec_with_default(specs, "CodeBound", None)
    if code_bound is not None:
        code_bound = float(code_bound)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    eikonal_lambda = float(get_spec_with_default(specs, "EikonalLambda", 0.0))

    # Trajectory datasets (generate_primitive_dataset, frames_per_trajectory > 1):
    #   TrajectorySmoothnessLambda: second difference of the control points
    #     along each trajectory -> linear latent interpolation follows the motion
    #   TrajectoryVelocityLambda: (df/dz) . dz/dt against the sampled dsdf/dt
    #     near the surface -> the latent Jacobian reproduces boundary velocities
    trajectory_smoothness_lambda = float(
        get_spec_with_default(specs, "TrajectorySmoothnessLambda", 0.0)
    )
    trajectory_velocity_lambda = float(
        get_spec_with_default(specs, "TrajectoryVelocityLambda", 0.0)
    )
    # near-surface points per scene and batch for the velocity term (None: all);
    # the term costs a second forward/backward with create_graph per point
    trajectory_velocity_points = get_spec_with_default(
        specs, "TrajectoryVelocityPoints", None
    )

    loss_type = get_spec_with_default(specs, "LossType", "ClampedL1")
    if loss_type.lower() == "clampedl1":
        loss_fn = ClampedL1Loss(clamp_val=clamp_dist)
    elif loss_type.lower() == "l1":
        loss_fn = torch.nn.L1Loss()
    elif loss_type.lower() == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown LossType: {loss_type}")

    train_split_file = pathlib.Path(data_source) / specs["TrainSplit"]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = DeepSDFStruct.deep_sdf.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        load_ram=True,
        geom_dimension=geom_dimension,
    )
    num_scenes = len(sdf_dataset)
    logging.info(f"There are {num_scenes} scenes")

    use_trajectories = (
        trajectory_smoothness_lambda > 0 or trajectory_velocity_lambda > 0
    )
    trajectories = {}
    scene_to_frame = {}  # scene index -> (trajectory id, position in trajectory)
    if use_trajectories:
        trajectories = DeepSDFStruct.deep_sdf.data.read_trajectory_info(
            data_source, sdf_dataset.npyfiles
        )
        if not trajectories:
            raise ValueError(
                "TrajectorySmoothnessLambda / TrajectoryVelocityLambda need a "
                "trajectory dataset (npz files with trajectory_id / frame_index / t)"
            )
        for traj, frames in trajectories.items():
            for position, (sid, _) in enumerate(frames):
                scene_to_frame[sid] = (traj, position)
        logging.info(
            f"{len(trajectories)} trajectories, "
            f"{len(scene_to_frame)}/{num_scenes} scenes are trajectory frames"
        )

    num_data_loader_threads = int(get_spec_with_default(specs, "DataLoaderThreads", 1))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )

    # spline latent fields (one per scene)
    latent_fields = make_latent_fields(
        num_scenes=num_scenes,
        latent_dim=latent_dim,
        tiling=tiling,
        bounds=bounds_param_space,
        device=device,
        init_std=spline_init_std,
        degrees=spline_degrees,
    )

    # build one lattice-struct per scene (shares decoder via model)
    # DeepSDFModel needs a "trained_latent_vectors" tensor just to define latent_dim.
    dummy_latents = torch.zeros((1, latent_dim), device=device, dtype=torch.float32)
    deep_sdf_model = DeepSDFModel(decoder, dummy_latents, device=device)

    structs = torch.nn.ModuleList()
    for sid in range(num_scenes):
        microtile = SDFfromDeepSDF(deep_sdf_model)
        struct = LatticeSDFStruct(
            tiling=tiling,
            microtile=microtile,
            parametrization=latent_fields[sid],
            bounds=bounds_param_space,
        )
        structs.append(struct)

    # decodes latents that are not a spline evaluation (velocity term)
    probe_latents = _FixedLatents()
    probe_struct = LatticeSDFStruct(
        tiling=tiling,
        microtile=SDFfromDeepSDF(deep_sdf_model),
        parametrization=probe_latents,
        bounds=bounds_param_space,
    )

    lr_schedules_spec = get_spec_with_default(
        specs,
        "LearningRateSchedule",
        [
            {
                "Type": "Step",
                "Initial": 1e-4,
                "Interval": 100,
                "Factor": 0.5,
            },  # decoder
            {
                "Type": "Step",
                "Initial": 1e-3,
                "Interval": 100,
                "Factor": 0.5,
            },  # spline CPs
        ],
    )
    if not isinstance(lr_schedules_spec, list) or len(lr_schedules_spec) != 2:
        raise ValueError(
            "LearningRateSchedule must be a list of two schedules: [decoder, latent_fields]"
        )
    lr_schedules = [_make_lr_schedule(s) for s in lr_schedules_spec]

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": latent_fields.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    def adjust_learning_rate(epoch):
        for i, param_group in enumerate(optimizer_all.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    start_epoch = 1
    loss_log = []
    lr_log = []
    timing_log = []
    lat_mag_log = []
    param_mag_log = {}

    if continue_from is not None:
        # Normalize checkpoint name: ws functions expect name without .pth,
        # local save/load functions expect the full filename.
        ckpt_filename = (
            continue_from if continue_from.endswith(".pth") else continue_from + ".pth"
        )
        ckpt_label = ckpt_filename[:-4]

        try:
            model_epoch = ws.load_model_parameters(
                experiment_directory, ckpt_label, decoder, device=device
            )
        except Exception:
            model_epoch = 1

        try:
            _ = load_optimizer(experiment_directory, ckpt_filename, optimizer_all)
        except Exception:
            pass

        try:
            _ = load_latent_fields(
                experiment_directory, ckpt_filename, latent_fields, device=device
            )
        except Exception:
            pass

        try:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = (
                load_logs(experiment_directory)
            )
            if log_epoch != model_epoch:
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                    loss_log,
                    lr_log,
                    timing_log,
                    lat_mag_log,
                    param_mag_log,
                    model_epoch,
                )
        except Exception:
            logging.warning("Could not load logs, starting fresh log history.")

        start_epoch = model_epoch + 1

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    logging.info("Starting training")
    decoder.train()
    latent_fields.train()

    start_train = time.time()
    global_step = 0
    error = 0.0
    total_time = "0:00:00"
    # The loop body never runs when resuming a run that already reached
    # NumEpochs, and the summary below reads `epoch`. Seed it with the last
    # completed epoch so that case returns a summary instead of raising
    # UnboundLocalError (`error` and `total_time` above are pre-set for the
    # same reason).
    epoch = start_epoch - 1
    for epoch in range(start_epoch, num_epochs + 1):
        start = time.time()
        adjust_learning_rate(epoch)

        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_traj_smooth = 0.0
        epoch_traj_vel = 0.0
        n_batches = 0
        epoch_error = 0.0

        for sdf_data, properties, indices in sdf_loader:
            # sdf_data: (ScenesPerBatch, SamplesPerScene, geom_dim+1), plus a
            # dsdf/dt column in trajectory datasets
            sdf_data = sdf_data.reshape(-1, sdf_data.shape[-1]).to(device)
            indices = indices.to(device)  # (ScenesPerBatch,)

            xyz = sdf_data[:, 0:geom_dimension]
            sdf_gt = sdf_data[:, geom_dimension].unsqueeze(1)
            if trajectory_velocity_lambda > 0:
                if sdf_data.shape[1] < geom_dimension + 2:
                    raise ValueError(
                        "TrajectoryVelocityLambda needs the dsdf/dt sample column"
                    )
                dsdf_dt_chunks = torch.chunk(
                    sdf_data[:, geom_dimension + 1], batch_split
                )

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz_chunks = torch.chunk(xyz, batch_split)
            sdf_chunks = torch.chunk(sdf_gt, batch_split)

            # expand scene indices per sample
            # indices: (ScenesPerBatch,) -> (ScenesPerBatch*SamplesPerScene,)
            expanded_idx = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)
            idx_chunks = torch.chunk(expanded_idx, batch_split)

            optimizer_all.zero_grad(set_to_none=True)

            batch_loss = 0.0
            batch_reg = 0.0
            batch_traj_smooth = 0.0
            batch_traj_vel = 0.0

            for i in range(batch_split):
                xyz_i = xyz_chunks[i]
                sdf_i = sdf_chunks[i]
                idx_i = idx_chunks[i]

                # compute predictions per scene id (supports ScenesPerBatch > 1)
                pred = torch.zeros_like(sdf_i)
                mask_cache = {}

                # IMPORTANT: struct internally
                #   xyz -> normalize by bounds -> spline -> microtile._set_param -> decode
                for sid in idx_i.unique():
                    sid_int = int(sid.item())
                    mask = idx_i == sid
                    mask_cache[sid_int] = mask
                    pred[mask] = structs[sid_int](xyz_i[mask])

                if enforce_minmax:
                    pred = torch.clamp(pred, minT, maxT)

                loss = loss_fn(pred, sdf_i)

                # L2 regularization on latent codes
                reg = 0.0
                n_reg_terms = 0
                for sid in idx_i.unique():
                    sid_int = int(sid.item())
                    if code_reg_target in ("evaluated", "both"):
                        # Regularize actual latent codes at query positions
                        # (what the decoder sees)
                        samples_clamped = xyz_i[mask_cache[sid_int]].clamp(
                            bounds_param_space[0], bounds_param_space[1]
                        )
                        lat_codes = latent_fields[sid_int](samples_clamped)
                        reg = reg + lat_codes.pow(2).mean()
                        n_reg_terms += 1
                    if code_reg_target in ("control_points", "both"):
                        # Regularize raw spline control points
                        for p in latent_fields[sid_int].parameters():
                            reg = reg + p.pow(2).mean()
                            n_reg_terms += 1

                reg = reg / max(n_reg_terms, 1)
                warmup = min(1.0, epoch / 100.0)
                loss_total = loss + code_reg_lambda * warmup * reg

                # Eikonal regularization: enforce ||grad SDF|| ~ 1 near surface
                if eikonal_lambda > 0:
                    near_mask = sdf_i.abs().squeeze() < clamp_dist * 0.5
                    if near_mask.any():
                        # Evaluate each scene's near-surface points with its own
                        # latent field. Decoding all near points through a single
                        # scene's struct would apply the wrong latent field to the
                        # other scenes' points (incorrect when ScenesPerBatch > 1).
                        idx_near = idx_i[near_mask]
                        xyz_near = xyz_i[near_mask]
                        eik_sq_residuals = []
                        for sid in idx_near.unique():
                            sid_int = int(sid.item())
                            sid_mask = idx_near == sid
                            xyz_eik = (
                                xyz_near[sid_mask].detach().clone().requires_grad_(True)
                            )
                            pred_eik = structs[sid_int](xyz_eik)
                            grad_sdf = torch.autograd.grad(
                                pred_eik.sum(), xyz_eik, create_graph=True
                            )[0]
                            eik_sq_residuals.append((grad_sdf.norm(dim=-1) - 1) ** 2)
                        eikonal_loss = torch.cat(eik_sq_residuals).mean()
                        loss_total = loss_total + eikonal_lambda * eikonal_loss

                # Trajectory velocity: the decoded SDF must change in time like
                # the ground truth, d f(x, z(x,t))/dt = dsdf/dt, near the surface
                if trajectory_velocity_lambda > 0:
                    inside = (
                        (xyz_i >= bounds_param_space[0])
                        & (xyz_i <= bounds_param_space[1])
                    ).all(dim=-1)
                    near = (sdf_i.abs().squeeze(-1) < clamp_dist) & inside
                    vel_residuals = []
                    for sid in idx_i.unique():
                        sid_int = int(sid.item())
                        if sid_int not in scene_to_frame:
                            continue
                        mask = mask_cache[sid_int] & near
                        if not mask.any():
                            continue
                        traj, position = scene_to_frame[sid_int]
                        sel = mask.nonzero(as_tuple=True)[0]
                        if trajectory_velocity_points is not None and sel.numel() > int(
                            trajectory_velocity_points
                        ):
                            perm = torch.randperm(sel.numel(), device=sel.device)
                            sel = sel[perm[: int(trajectory_velocity_points)]]
                        xyz_v = xyz_i[sel]
                        z = latent_fields[sid_int](xyz_v)
                        z_dot = latent_time_derivative(
                            latent_fields, trajectories[traj], position, xyz_v
                        )
                        dfdt = decoded_time_derivative(
                            probe_struct, probe_latents, z, z_dot, xyz_v
                        )
                        vel_residuals.append((dfdt - dsdf_dt_chunks[i][sel]).abs())
                    if vel_residuals:
                        vel_loss = torch.cat(vel_residuals).mean()
                        loss_total = loss_total + trajectory_velocity_lambda * vel_loss
                        batch_traj_vel += float(vel_loss.detach().item()) / batch_split

                # Normalize by batch_split so the accumulated gradient (and the
                # logged loss) equal the full-batch mean regardless of BatchSplit.
                loss_total = loss_total / batch_split
                loss_total.backward()

                batch_loss += float(loss.detach().item()) / batch_split
                batch_reg += float(reg.detach().item()) / batch_split

            # Trajectory smoothness on the control points of every trajectory
            # this batch touches (all its frames, so the whole path is shaped)
            if trajectory_smoothness_lambda > 0:
                touched = {
                    scene_to_frame[int(sid)][0]
                    for sid in indices.unique().tolist()
                    if int(sid) in scene_to_frame
                }
                smooth = trajectory_smoothness(
                    latent_fields, [trajectories[traj] for traj in sorted(touched)]
                )
                if smooth is not None:
                    (trajectory_smoothness_lambda * smooth).backward()
                    batch_traj_smooth = float(smooth.detach().item())

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(latent_fields.parameters(), grad_clip)

            optimizer_all.step()

            if code_bound is not None:
                with torch.no_grad():
                    for lf in latent_fields:
                        for p in lf.parameters():
                            p.clamp_(-code_bound, code_bound)

            epoch_error += batch_loss
            if use_mlflow and (global_step % mlflow_log_every_n_batches == 0):
                # log batch-level metrics
                batch_metrics = {
                    "batch_loss": float(batch_loss),
                    "batch_reg": float(batch_reg),
                }
                if trajectory_smoothness_lambda > 0:
                    batch_metrics["batch_traj_smooth"] = batch_traj_smooth
                if trajectory_velocity_lambda > 0:
                    batch_metrics["batch_traj_vel"] = batch_traj_vel
                mlflow.log_metrics(batch_metrics, step=global_step)
            global_step += 1

            loss_log.append(float(batch_loss))

            epoch_loss += batch_loss
            epoch_reg += batch_reg
            epoch_traj_smooth += batch_traj_smooth
            epoch_traj_vel += batch_traj_vel
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_reg = epoch_reg / max(1, n_batches)
        avg_traj_smooth = epoch_traj_smooth / max(1, n_batches)
        avg_traj_vel = epoch_traj_vel / max(1, n_batches)
        traj_string = ""
        if trajectory_smoothness_lambda > 0:
            traj_string += f"Traj. smooth: {avg_traj_smooth:.3e} "
        if trajectory_velocity_lambda > 0:
            traj_string += f"Traj. vel.: {avg_traj_vel:.4f} "

        error = epoch_error / len(sdf_loader)
        tot_time = time.time() - start_train
        avg_time_per_epoch = tot_time / (epoch)
        estimated_remaining_time = avg_time_per_epoch * (num_epochs - (epoch))
        time_string = str(datetime.timedelta(seconds=round(estimated_remaining_time)))

        if epoch == num_epochs:
            total_time = str(datetime.timedelta(seconds=round(tot_time)))
            logging.info(
                f"Finished {epoch} ({epoch}/{num_epochs}) [{epoch/num_epochs*100:.2f}%] after {total_time}"
            )
        else:
            logging.info(
                f"Finished epoch {epoch:5g}/{num_epochs} | "
                f"with Reg.: {avg_reg:.4f} "
                f"{traj_string}"
                f"and Tot.: {avg_loss:.4f} "
                f"[{epoch/num_epochs*100:.2f}%] in {time_string} "
                f"({avg_time_per_epoch:.2f}s/epoch)"
            )

        timing_log.append(time.time() - start)

        lr_log.append(
            [lr_schedules[i].get_learning_rate(epoch) for i in range(len(lr_schedules))]
        )

        lat_mag_log.append(get_mean_spline_param_magnitude(latent_fields))

        append_parameter_magnitudes(param_mag_log, decoder)

        if use_mlflow:
            epoch_metrics = {
                "epoch_loss": float(avg_loss),
                "epoch_reg": float(avg_reg),
                "latent_field_mean_param_norm": float(lat_mag_log[-1]),
                "lr_decoder": float(optimizer_all.param_groups[0]["lr"]),
                "lr_latent_fields": float(optimizer_all.param_groups[1]["lr"]),
            }
            if trajectory_smoothness_lambda > 0:
                epoch_metrics["epoch_traj_smooth"] = float(avg_traj_smooth)
            if trajectory_velocity_lambda > 0:
                epoch_metrics["epoch_traj_vel"] = float(avg_traj_vel)
            mlflow.log_metrics(epoch_metrics, step=int(epoch))

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )

        if use_mlflow and (epoch % log_frequency == 0):
            # logs + plot
            logs_path = os.path.join(experiment_directory, ws.logs_filename)
            logplot_path = os.path.join(experiment_directory, ws.logplot_filename)
            if os.path.isfile(logs_path):
                mlflow.log_artifact(logs_path)
            if os.path.isfile(logplot_path):
                mlflow.log_artifact(logplot_path)

            # checkpoints you just saved
            model_params_dir = ws.get_model_params_dir(
                experiment_directory, create_if_nonexistent=False
            )
            optim_dir = ws.get_optimizer_params_dir(
                experiment_directory, create_if_nonexistent=False
            )
            lat_dir = ws.get_latent_codes_dir(
                experiment_directory, create_if_nonexistent=False
            )

            for p in [
                os.path.join(model_params_dir, "latest.pth"),
                os.path.join(optim_dir, "latest.pth"),
                os.path.join(lat_dir, "latest.pth"),
            ]:
                if os.path.isfile(p):
                    mlflow.log_artifact(p)

    summary = ws.ExperimentSummary(
        loss=error,
        num_epochs=epoch,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        host_name=host_name,
        device=str(device),
        training_duration=total_time,
        data_dir=str(data_source),
        version=version("DeepSDFStruct"),
    )
    ws.save_experiment_summary(experiment_directory, summary)

    return summary


def export_training_latent_fields_to_stl(
    experiment_directory: str,
    checkpoint: str = "latest.pth",
    out_dir: str | None = None,
    N_base: int = 12,
    device: str | None = None,
    overwrite: bool = True,
    max_scenes: int = 3,
):
    """
    Export one STL per training scene using the *trained spline latent fields* (control points),
    and the trained decoder weights at `checkpoint`.

    Uses the same geometry pipeline as test_reconstruction.py:
      SDFfromDeepSDF -> LatticeSDFStruct -> create_3D_mesh -> export_surface_mesh
    """
    experiment_directory = str(experiment_directory)
    specs = ws.load_experiment_specifications(experiment_directory)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # where to save
    if out_dir is None:
        out_dir = os.path.join(experiment_directory, "reconstructions_latent_field")
    os.makedirs(out_dir, exist_ok=True)

    # load decoder checkpoint
    # ws.load_trained_model expects the checkpoint without ".pth" (e.g. "latest"),
    # while save_latent_fields writes filenames like "latest.pth" — accept both.
    ckpt_name = checkpoint
    ckpt_noext = checkpoint[:-4] if checkpoint.endswith(".pth") else checkpoint

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
    if len(tiling) != geom_dimension:
        raise ValueError(
            f"Tiling length must match geom_dimension={geom_dimension}. Got {tiling}"
        )

    bounds_param_space = get_spec_with_default(
        specs, "BoundsParamSpace", [[-1.0] * geom_dimension, [1.0] * geom_dimension]
    )
    bounds_param_space_t = torch.tensor(
        bounds_param_space, dtype=torch.float32, device=device
    )

    spline_init_std = float(get_spec_with_default(specs, "SplineInitStd", 0.01))
    spline_degrees = get_spec_with_default(specs, "SplineDegrees", None)

    # determine number of scenes from the training split (same as in train())
    data_source = specs["DataSource"]
    train_split_file = pathlib.Path(data_source) / specs["TrainSplit"]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    # We only need the dataset length here: SDFSamples.__len__ is the scene count.
    sdf_dataset = SDFSamples(
        data_source,
        train_split,
        subsample=1,  # not used here
        load_ram=False,  # we won't draw samples here
        geom_dimension=geom_dimension,
    )
    num_scenes = len(sdf_dataset)

    # build latent fields container and load CPs
    latent_fields = make_latent_fields(
        num_scenes=num_scenes,
        latent_dim=latent_dim,
        tiling=tiling,
        bounds=bounds_param_space_t,
        device=device,
        init_std=spline_init_std,
        degrees=spline_degrees,
    )
    _ = load_latent_fields(
        experiment_directory, ckpt_name, latent_fields, device=device
    )
    latent_fields.eval()

    # DeepSDFModel needs a tensor library just to define latent_dim
    dummy_latents = torch.zeros((1, latent_dim), device=device, dtype=torch.float32)
    deep_sdf_model = DeepSDFModel(decoder, dummy_latents, device=device)
    microtile = SDFfromDeepSDF(deep_sdf_model)

    # export
    for sid in range(num_scenes):
        if sid == max_scenes:
            print(f"Reached max_scenes={max_scenes}, stopping.")
            break
        out_path = os.path.join(out_dir, f"scene_{sid:05d}.stl")
        if (not overwrite) and os.path.isfile(out_path):
            print(f"[skip] {out_path}")
            continue

        struct = LatticeSDFStruct(
            tiling=tiling,
            microtile=microtile,
            parametrization=latent_fields[sid],
            bounds=bounds_param_space,
        )

        # Create mesh in param space bounds (same pattern as test_reconstruction.py)
        surf_mesh, derivative = create_3D_mesh(
            struct,
            int(N_base),
            differentiate=False,
            device=device,
            mesh_type="surface",
            bounds=bounds_param_space_t,
            deformation_function=None,
        )

        export_surface_mesh(out_path, surf_mesh.to_gus(), derivative)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    experiment_dir = "DeepSDFStruct/trained_models/primitives_cl32"
    train(experiment_dir)
    export_training_latent_fields_to_stl(experiment_dir, checkpoint="latest.pth")
