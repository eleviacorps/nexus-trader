# Evaluation Entrypoints

Canonical evaluation scripts moved here for cleaner structure.

## Available Entrypoints

- `evaluate_branch_accuracy.py`
- `evaluate_branch_realism.py`
- `evaluate_diffusion_v2.py`
- `evaluate_v12_tctl.py`
- `evaluate_v13_cabr.py`
- `evaluate_v24_system.py`
- `evaluate_v26_phase1.py`
- `evaluate_v26_phase2.py`
- `evaluate_v26_phase2b.py`
- `evaluate_v26_phase2_advanced.py`
- `evaluate_v27_15min.py`

## Backward Compatibility

You can still run commands like `python scripts/evaluate_v26_phase2.py`.  
Those files are wrappers that delegate to this directory.
