# Project Guidelines

## Build and Test
- Install dependencies (preferred): `uv sync`
- Run tests (CI): `uv run pytest --cov=DeepSDFStruct --cov-report=term-missing --cov-report=xml tests`
- Run a single test file: `uv run pytest tests/test_sdf_functions.py`
- Run a single test: `uv run pytest tests/test_sdf_functions.py::test_name`
- Build docs (CI):
  - `uv sync --extra docs --no-editable`
  - `uv run sphinx-autogen -o docs/source/generated docs/api_reference.rst`
  - `uv run make -C docs html`

## Code Style
- Python formatting: `black DeepSDFStruct/ tests/` (also via `.pre-commit-config.yaml`).

## Architecture
- High-level architecture and module responsibilities: see `CLAUDE.md`.
- Core modules you’ll touch most:
  - `DeepSDFStruct/SDF.py`: `SDFBase` (`torch.nn.Module`) abstraction.
  - `DeepSDFStruct/lattice_structure.py`: periodic tiling + optional spline deformation.
  - `DeepSDFStruct/torch_spline.py`: differentiable B-splines (wraps `splinepy`).
  - `DeepSDFStruct/sampling.py`: sampling points/distances from SDFs.
  - `DeepSDFStruct/mesh.py`: FlexiCubes/FlexiSquares extraction + export.
  - `DeepSDFStruct/deep_sdf/`: DeepSDF training/reconstruction + metrics.

## Conventions
- Prefer `uv` workflows: this repo uses git-sourced dependencies (e.g. `splinepy`, `gustaf`) configured in `pyproject.toml`.
- Device/dtype hygiene:
  - Create new tensors on the SDF’s device/dtype (use `sdf.get_device()` / `sdf.get_dtype()` patterns where available).
  - Keep GPU tensors out of NumPy/mesh IO code paths; detach and move to CPU only at boundaries.
- Differentiability:
  - Avoid accidental `.detach()` in code paths that are meant to remain differentiable.
  - Only detach/copy when interfacing with non-PyTorch libraries (export, logging, external solvers).
- SDF I/O shapes:
  - Query points are typically shaped `(N, 3)` (or `(N, 2)` for 2D).
  - SDF evaluations are expected to return `(N, 1)`.
