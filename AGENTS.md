# Repository Guidelines

## Project Structure & Module Organization
Core training code lives in `pretrain.py` and pulls architectures from `models/` and `config/arch/`. Dataset builders and augmentation scripts sit in `dataset/`, while evaluation routines and metrics are under `evaluators/`. Shared utilities (logging, scheduling, helpers) reside in `utils/`. The `gpt-integration/` folder contains agent tooling and its tests, and `assets/` stores figures referenced by documentation. Keep new experiments and configs grouped by domain to maintain discoverability.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies after creating a 3.10 environment.
- `python -m dataset.build_arc_dataset --help`: inspect dataset generation flags before running large builds.
- `torchrun --nproc-per-node 4 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" ...`: launch distributed training; mirror the README templates for other puzzles.
- `python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]"`: run single-GPU or CPU debugging sweeps.

## Coding Style & Naming Conventions
Follow standard Python formatting with 4-space indentation and type hints where practical. Use snake_case for functions and variables, PascalCase for classes, and hyphenated config names (e.g., `cfg_pretrain.yaml`). Keep Hydra/OmegaConf configs declarative; prefer extending existing YAML files over in-line overrides sprinkled throughout scripts. Document non-obvious logic with concise comments and maintain docstrings for public APIs.

## Testing Guidelines
Use `pytest gpt-integration/tests` for agent regressions and add new suites alongside the feature they exercise. Provide minimal, deterministic fixtures; large fixtures belong in `dataset/fixtures/` when shared. For training changes, include a short smoke script (e.g., `python pretrain.py epochs=1 evaluators="[]"`) in the PR description showing the run completes. Consider adding metric assertions when modifying evaluators to avoid silent regressions.

## Commit & Pull Request Guidelines
Write commits in the imperative mood (e.g., “Add TRM ablation utilities”) and keep related changes squashed. Reference issues or experiment IDs in the body when applicable. Pull requests should summarize scope, list key commands executed, and attach W&B links or loss curves for training work. Request review only after linting/testing passes and configuration files load via Hydra without runtime overrides errors.

## Experiment Tracking & Configuration Tips
Store new configs in `config/` with self-descriptive filenames and note expected hardware in comments. Avoid committing raw datasets or checkpoints; instead, document download or generation instructions. When using Weights & Biases, rely on environment variables or local profiles rather than checking in secrets.
