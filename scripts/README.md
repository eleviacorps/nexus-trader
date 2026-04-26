# Scripts Layout

This directory contains executable workflows grouped by purpose.

## Categories

- `evaluate_*.py` wrappers for compatibility; canonical evaluation entrypoints now live under `scripts/evaluation/`
- `train_*.py` model training jobs
- `run_*.py` orchestration and experiment runners
- `validate_*.py` validation and consistency checks
- `sync_*.py` local/remote sync utilities
- `test_*.py` ad-hoc test harnesses

## Evaluation Scripts

Canonical paths are in `scripts/evaluation/`. Backward-compatible wrappers remain at `scripts/evaluate_*.py`.

Examples:

- `python scripts/evaluate_v26_phase2.py`
- `python scripts/evaluate_v26_phase2_advanced.py`
- `python scripts/evaluate_branch_accuracy.py`
