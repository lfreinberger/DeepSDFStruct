#!/usr/bin/env python3
"""
Random-search hyperparameter tuning for the latent-field decoder.
=================================================================

Wraps :func:`DeepSDFStruct.deep_sdf.training_latent_field.train` (the latent-field /
spline trainer used by e.g. ``primitives_cl32_new``) in a random search over the
hyperparameters that matter for this pipeline:

  - ``CodeLength``            latent dimension {8, 16, 32}
  - ``Tiling``               latent control-lattice resolution
  - learning rates + reg.    decoder/spline LR, CodeRegularizationLambda,
                             SplineInitStd, EikonalLambda
  - network architecture     decoder width/depth (``dims``), dropout

Each trial samples a configuration, writes a ``specs.json`` into an MLflow run's
artifact directory, trains for a *reduced* number of epochs, and records the best
held-out **validation loss** (the generalization metric the trainer already
computes). After the search, trials are ranked and the best override set is saved;
full-train the winner with the normal ``train()`` entry point.

Run directly with the editable ``CONFIG`` dict at the bottom::

    python -m DeepSDFStruct.deep_sdf.hyperparameter_search

Inspect results with the MLflow UI::

    mlflow ui --backend-store-uri <tracking_uri>
"""

import json
import logging
import pathlib
from urllib.parse import urlparse

import numpy as np
import mlflow

import DeepSDFStruct
from DeepSDFStruct.design_of_experiments import (
    ExperimentSpecifications,
    create_experiment,
)
from DeepSDFStruct.deep_sdf.training_latent_field import train, load_logs

logger = logging.getLogger(DeepSDFStruct.__name__)


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))


def sample_overrides(rng: np.random.Generator, groups: list[str]) -> dict:
    """Draw one random specs-override dict for the enabled parameter groups."""
    ov: dict = {}

    if "code_length" in groups:
        ov["CodeLength"] = int(rng.choice([8, 16, 32]))

    if "tiling" in groups:
        tilings = [[4, 4, 4], [6, 6, 6], [8, 8, 8], [12, 12, 12]]
        ov["Tiling"] = list(tilings[int(rng.integers(len(tilings)))])

    if "learning" in groups:
        dec_lr = _loguniform(rng, 1e-4, 3e-3)
        spl_lr = _loguniform(rng, 1e-4, 3e-3)
        # keep Step schedule; large interval => ~constant over a short sweep run
        ov["LearningRateSchedule"] = [
            {"Type": "Step", "Initial": dec_lr, "Interval": 12000, "Factor": 0.5},
            {"Type": "Step", "Initial": spl_lr, "Interval": 12000, "Factor": 0.5},
        ]
        ov["CodeRegularizationLambda"] = _loguniform(rng, 1e-5, 1e-3)
        ov["SplineInitStd"] = _loguniform(rng, 1e-3, 1e-1)
        ov["EikonalLambda"] = float(rng.choice([0.0, 0.0, 0.01, 0.1]))

    if "architecture" in groups:
        width = int(rng.choice([64, 128, 256]))
        depth = int(rng.choice([3, 4, 5, 6]))
        idx = list(range(depth + 1))  # cover all hidden layers (extra indices are no-ops)
        ov["NetworkSpecs"] = {
            "dims": [width] * depth,
            "dropout": idx,
            "norm_layers": idx,
            "dropout_prob": float(rng.choice([0.0, 0.1, 0.2])),
        }

    return ov


def _best_val_loss(exp_dir: str, fallback: float) -> float:
    """Minimum held-out validation loss recorded during the run."""
    try:
        _, _, _, _, _, val_log, _ = load_logs(exp_dir)
    except Exception:
        val_log = []
    return min((float(v) for _, v in val_log), default=float(fallback))


def run_random_search(cfg: dict) -> list[dict]:
    """Run the random search defined by ``cfg`` and return ranked trial results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    base = ExperimentSpecifications(cfg["base_specs"])
    rng = np.random.default_rng(cfg["seed"])

    # reduced per-trial budget (always overrides whatever was sampled)
    reduced = int(cfg["reduced_epochs"])
    log_freq = int(cfg.get("log_frequency", max(1, reduced // 8)))
    budget = {
        "NumEpochs": reduced,
        "LogFrequency": log_freq,
        "ValidationFrequency": int(cfg.get("validation_frequency", log_freq)),
        "SnapshotFrequency": reduced,
        "AdditionalSnapshots": [],
    }
    budget.update(cfg.get("extra_overrides", {}))

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["sweep_name"])

    results: list[dict] = []
    for t in range(int(cfg["n_trials"])):
        specs = base.copy()
        ov = sample_overrides(rng, list(cfg["groups"]))
        specs.update(ov)
        specs.update(budget)

        with mlflow.start_run(run_name=f"trial_{t:03d}"):
            parsed = urlparse(mlflow.get_artifact_uri())
            if parsed.path == "":
                raise NotImplementedError("Remote tracking URI not supported.")
            exp_dir = parsed.path

            create_experiment(exp_dir, specs=dict(specs))
            mlflow.log_param("trial", t)
            mlflow.log_params(ExperimentSpecifications(ov).flatten(parent_key="search"))

            try:
                summary = train(
                    exp_dir,
                    device=cfg.get("device"),
                    use_mlflow=True,
                    mlflow_run_name=f"trial_{t:03d}",
                )
            except Exception:
                logger.exception(f"trial {t} failed")
                mlflow.set_tag("status", "failed")
                continue

            fallback = (
                summary["loss"] if isinstance(summary, dict) else summary.loss
            )
            best_val = _best_val_loss(exp_dir, fallback=fallback)
            mlflow.log_metric("best_val_loss", best_val)
            results.append(
                {"trial": t, "best_val_loss": best_val, "overrides": ov}
            )
            logger.info(f"trial {t}: best_val_loss={best_val:.5f} | {ov}")

    results.sort(key=lambda r: r["best_val_loss"])

    out_dir = pathlib.Path(cfg.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['sweep_name']}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"\n=== Top {min(5, len(results))} configs (by val loss) ===")
    for r in results[:5]:
        logger.info(f"  val={r['best_val_loss']:.5f}  {r['overrides']}")
    logger.info(f"Saved full results to {out_path}")
    if results:
        logger.info(
            "Full-train the winner by applying its overrides to the base "
            "specs.json and running train() with the normal NumEpochs."
        )

    return results


CONFIG = {
    "base_specs": "DeepSDFStruct/trained_models/primitives_cl32_new/specs.json",
    "sweep_name": "primitives_cl32_random_search",
    "groups": ["code_length", "tiling", "learning", "architecture"],
    "n_trials": 20,
    "reduced_epochs": 400,  # per-trial budget (full run is 10000)
    "log_frequency": 50,
    "validation_frequency": 50,
    "device": None,  # None -> cuda if available else cpu
    "tracking_uri": "mlruns",
    "output_dir": "DeepSDFStruct/trained_models/primitives_cl32_new",
    "seed": 0,
    # extra specs overrides applied to every trial (e.g. shrink for a quick test)
    "extra_overrides": {},
}


if __name__ == "__main__":
    run_random_search(CONFIG)
