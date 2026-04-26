# Nexus Trader

Nexus Trader is a research + execution codebase for market simulation, path generation, and trading-system evaluation.

The project spans:

- data preparation and feature fusion
- branch/path generation (including diffusion-based stacks)
- ranking, realism scoring, and regime-aware evaluation
- packaged runtime (`nexus_packaged/`) for service-style execution

## Core Architecture

`WORLD -> PERCEPTION -> SIMULATION -> FUTURE BRANCHING -> REVERSE COLLAPSE -> PROBABILITY CONE`

## Cleaned Repository Layout

```text
nexus-trader/
  config/                  project paths and runtime configuration
  src/                     core Python source modules
  scripts/                 runnable jobs (train/run/evaluate/validate/sync)
    evaluation/            canonical evaluation entrypoints
  docs/                    active documentation
  nexus_old/               archived legacy docs/scripts/artifacts
  nexus_packaged/          packaged runtime stack and deployment-oriented modules
  data/, outputs/, models/ local artifacts (mostly ignored from git history)
```

## Prerequisites

- Python 3.10+ (recommended 3.11)
- `pip`
- Optional GPU runtime for training/eval acceleration (CUDA/ROCm depending on your setup)

## Install

### 1. Create and activate virtual environment

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements-prod.txt
```

## Configuration

Main config surface:

- [`config/project_config.py`](config/project_config.py)

Packaged runtime settings:

- [`nexus_packaged/config/settings.json`](nexus_packaged/config/settings.json)
- optional local env file: `nexus_packaged/.env.local`

## Common Workflows

### 1. Build/refresh fused features

```bash
python scripts/build_fused_artifacts.py
```

### 2. Train core model stacks

Examples:

```bash
python scripts/train_fused_tft.py --epochs 2 --batch-size 512 --sequence-len 180 --hidden-dim 192 --dropout 0.15 --sample-limit 1000000
python scripts/train_v24_diffusion_v2.py
python scripts/train_v26_regime_diffusion.py
python scripts/train_v26_multi_horizon.py
```

### 3. Run evaluations

Canonical evaluation files are in `scripts/evaluation/`.

Examples:

```bash
python scripts/evaluate_branch_accuracy.py
python scripts/evaluate_diffusion_v2.py
python scripts/evaluate_v26_phase1.py
python scripts/evaluate_v26_phase2.py
python scripts/evaluate_v26_phase2_advanced.py
```

Note:

- Backward-compatible wrappers still exist at `scripts/evaluate_*.py`.
- Canonical implementations now live at `scripts/evaluation/evaluate_*.py`.

### 4. Run packaged runtime

```bash
python main.py
```

Equivalent direct entry:

```bash
python -m nexus_packaged.main
```

## Script Organization

See:

- [`scripts/README.md`](scripts/README.md)
- [`scripts/evaluation/README.md`](scripts/evaluation/README.md)

## File Location Database (for Future Agents)

This repo now includes an SQLite index workflow so agents can query file locations quickly instead of recursively searching every time.

Build or refresh index:

```bash
python scripts/build_file_index.py
```

Default output:

- `meta/file_index.sqlite`

Query examples:

```bash
python scripts/query_file_index.py --name evaluate_v26
python scripts/query_file_index.py --ext .py --category evaluation
python scripts/query_file_index.py --path nexus_packaged/v30 --limit 20
```

Details:

- [`docs/reference/FILE_INDEX.md`](docs/reference/FILE_INDEX.md)

## Documentation Map

- [`docs/README.md`](docs/README.md)
- [`docs/context/CONTEXT_HANDOFF.md`](docs/context/CONTEXT_HANDOFF.md)
- [`docs/operations/DEPLOYMENT.md`](docs/operations/DEPLOYMENT.md)
- [`docs/planning/MODEL_QUALITY_EXECUTION_PLAN.md`](docs/planning/MODEL_QUALITY_EXECUTION_PLAN.md)
- [`docs/reports/PROJECT_MASTER_SUMMARY.md`](docs/reports/PROJECT_MASTER_SUMMARY.md)
- [`nexus_old/README.md`](nexus_old/README.md)

## Archive Policy

- Active material stays under `src/`, `scripts/`, `docs/`, and `nexus_packaged/`.
- Legacy or one-off residue is moved to `nexus_old/`.
- Session dumps and old version notes are maintained in archive folders, not root.

## Notes

- The repository has been history-cleaned to remove large binary artifacts from git history.
- Keep large datasets/logs/models untracked and outside commit history unless intentionally versioned in a dedicated storage strategy.
