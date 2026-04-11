# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSDFStruct is a differentiable framework for designing and deforming 3D microstructured materials using Signed Distance Functions (SDFs), spline-based lattice structures, and neural shape encoding (DeepSDF). It targets topology optimization, additive manufacturing, and metamaterial research.

## Commands

### Setup and Install
```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Running Tests
```bash
# Run all tests with coverage
uv run pytest --cov=DeepSDFStruct --cov-report=term-missing tests/

# Run a single test file
uv run pytest tests/test_sdf_functions.py

# Run a single test
uv run pytest tests/test_sdf_functions.py::test_name
```

### Code Formatting
```bash
# Format with black (pre-commit hook)
black DeepSDFStruct/ tests/
```

### Building Docs
```bash
uv sync --extra docs --no-editable
uv run sphinx-autogen -o docs/source/generated docs/api_reference.rst
uv run make -C docs html
```

### CI
Tests run on Python 3.10–3.13 via GitHub Actions. Coverage is reported to Coveralls.

## Architecture

### Core Data Flow

```
Geometry (Mesh/Primitive)
  → SDF Representation (SDFBase subclass, all are torch.nn.Module)
  → Optional: LatticeSDFStruct (periodic tiling + spline deformation)
  → Optional: SplineParametrization (spatially-varying params, e.g. thickness)
  → Sampling (sampling.py) → Dataset → train DeepSDF model
  → Mesh Extraction: FlexiCubes (3D) / FlexiSquares (2D)
  → Export (VTK, PLY, STL, ONNX, MFEM)
  → Optional: MMA optimization with TorchFEM FEA in the loop
```

### Key Module Responsibilities

**`DeepSDFStruct/SDF.py`** — `SDFBase` abstract base (inherits `torch.nn.Module`). All geometry types subclass this: primitives (`sdf_primitives.py`), mesh-converted (`SDFfromMesh`), neural (`SDFfromDeepSDF`), boolean ops (union/intersection/difference), `LatticeSDFStruct`, `CappedBorderSDF`, `LocalShapesSDF`. The `forward()` method evaluates the SDF at query points.

**`DeepSDFStruct/lattice_structure.py`** — `LatticeSDFStruct` tiles a unit-cell SDF periodically in parametric space and optionally applies a spline deformation (via `TorchSpline`) to map to physical space. Parametrization (thickness/shape variation) is injected here.

**`DeepSDFStruct/mesh.py`** — Mesh extraction and export. `torchSurfMesh` / `torchVolumeMesh` store meshes as PyTorch tensors. `FlexiCubes` and `FlexiSquares` are state-of-the-art dual contouring algorithms that extract differentiable meshes from SDF grids. Export to VTK, Abaqus, PLY, MFEM.

**`DeepSDFStruct/sampling.py`** — Sampling points from SDFs for dataset generation. `SampledSDF` stores samples. Supports uniform, surface-focused, and importance sampling strategies. `DataSetInfo` manages dataset metadata.

**`DeepSDFStruct/torch_spline.py`** — Differentiable B-spline evaluation (`TorchSpline`, `torch_spline_1D/2D/3D`). Implements De Boor's algorithm in PyTorch. Wraps `splinepy` splines for use in gradient-based pipelines. `TorchScaling` handles coordinate normalization.

**`DeepSDFStruct/parametrization.py`** — `Constant` (uniform params) and `SplineParametrization` (smooth spatial variation of shape parameters). Used to define spatially-varying microstructure properties.

**`DeepSDFStruct/optimization.py`** — MMA (Method of Moving Asymptotes) optimizer wrapper. Integrates TorchFEM for FEA-based objectives (e.g., structural stiffness). Mesh quality utilities.

**`DeepSDFStruct/deep_sdf/`** — DeepSDF neural network components:
- `training.py` / `training_latent_field.py` — training pipelines with MLflow tracking
- `reconstruction.py` — reconstruct SDF from sparse samples using a trained model
- `networks/` — decoder architectures (`deep_sdf_decoder.py`, `quantum_deep_sdf_decoder.py` requires `uv sync --extra quantum`)
- `metrics/` — evaluation metrics including `mesh_to_analytical.py` and `error_metrics.py`
- `models.py` — `DeepSDFModel` wrapper (decoder + latent codes)

### Differentiability

All core operations preserve PyTorch autograd. SDFs are `nn.Module` subclasses; spline evaluations use torch ops; mesh extraction via FlexiCubes returns differentiable vertex positions. This enables end-to-end gradient flow from mesh quality / FEA objective back to SDF parameters or latent codes.

### Pretrained Models

Loaded via `pretrained_models.py` using `huggingface-hub`. Available as `PretrainedModels.AnalyticRoundCross` etc. Models are cached locally after first download.

### Experiment Tracking

MLflow is used for tracking training runs. The local `mlflow.db` SQLite database and `mlruns/` directory store experiment data. `DeepSDFStruct/design_of_experiments.py` provides `ExperimentSpecifications` for structured experiment management.

## Conventions

### Device / Dtype Hygiene
- Create new tensors on the SDF's device/dtype (use `sdf.get_device()` / `sdf.get_dtype()` patterns where available).
- Keep GPU tensors out of NumPy/mesh IO code paths; detach and move to CPU only at boundaries.

### Differentiability Guards
- Avoid accidental `.detach()` in code paths that are meant to remain differentiable.
- Only detach/copy when interfacing with non-PyTorch libraries (export, logging, external solvers).

### SDF I/O Shapes
- Query points: `(N, 3)` (or `(N, 2)` for 2D).
- SDF evaluations return: `(N, 1)`.

### Knot Vectors / Parametrization
Knot vectors are defined on the actual normalized domain (not a shifted/scaled version). See commit `be19b52` for context on this convention.
