# PROJECT_AUDIT

## Summary

The repository started as a notebook-first prototype with almost all reusable logic embedded directly inside notebook cells. The working code lived mainly in the old ABM and model notebooks, while most of the other notebook files were empty placeholders. There was no reusable `src/` implementation for the simulation, model, alignment, or validation layers.

This refactor moved the core architecture into modules under `src/`, introduced shared configuration in `config/project_config.py`, added validation/tests/scripts, and replaced the legacy notebook surfaces with thin numbered wrappers.

## Legacy Notebook Audit

### Old ABM notebook

This was the main ABM prototype.
It contained:
- Duplicated rule-based strategy definitions.
- Multiple strategy calibration passes in the same file.
- Persona config creation and JSON export.
- Single-step simulation logic.
- A hardcoded dependency on a legacy higher-timeframe feature CSV.

Problems found:
- Strategy code was redefined multiple times in separate cells.
- Persona definitions were notebook-local instead of module-backed.
- The narrative logic still described the older higher-timeframe behavior instead of the new 1-minute pipeline.

### Old branching notebook

This notebook only loaded the legacy feature file and persona config, but did not contain a complete reusable tree-search implementation.
It was effectively a stub.

Problems found:
- Hardcoded higher-timeframe feature paths.
- No separation between tree logic and visualization intent.
- No tests for leaf count or reverse collapse behavior.

### Old model notebook

This was the most overloaded notebook in the repo.
It contained:
- Strategy and persona logic duplicated from the ABM notebook.
- Simulation tree helpers duplicated from branching work.
- Normalization and dataset code.
- A notebook labelled as TFT that actually trained an LSTM-based model in core sections.
- GPU inspection commands embedded directly into the notebook.
- Multiple references to the legacy higher-timeframe historical feature file.

Problems found:
- The notebook mixed exploration, diagnostics, model definition, and training.
- `SEQUENCE_LEN` was still `60`, not the target `120`.
- The model implementation drifted away from the intended fused TFT architecture.
- Local device assumptions were wired directly into notebook cells.
- Feature-width assumptions were effectively price-only and not prepared for 36 + 32 + 32 fusion.

### Empty notebook files found before refactor

The following files existed but were empty placeholders:
- environment check
- data download
- feature engineering
- historical data
- news model
- crowd pipeline variants
- alternate TFT notebook

## Duplicated Logic And Constants

Duplicated or scattered pieces before refactor:
- Persona definitions appeared inside notebooks instead of shared modules.
- Strategy functions were copied across notebook cells.
- Simulation tree logic was redefined inside the training notebook.
- Path handling used notebook-relative hardcoded strings.
- Sequence length, split ratios, and batch sizing were not centralized.
- Feature subsets were implied by notebook cell ordering instead of explicit config.

## Broken Or Risky Assumptions Found

The old code still relied on or implied:
- Legacy higher-timeframe feature files for ABM and model work.
- A GPU-specific workflow embedded in notebooks.
- A price-only or partially price-only training surface.
- Notebook-local path resolution that would break between local Windows and remote ROCm server use.
- A mislabeled model path where “TFT” training code had drifted into an LSTM implementation.

## Refactor Delivered

New shared modules were created in:
- `config/project_config.py`
- `src/utils/`
- `src/data/window_dataset.py`
- `src/embeddings/`
- `src/models/`
- `src/training/train_tft.py`
- `src/simulation/`
- `src/mcts/`
- `scripts/`
- `tests/`

What now lives in modules:
- Shared paths, runtime detection, and feature config.
- 1-minute-calibrated strategy rules.
- Persona decision logic and config serialization.
- ABM one-step simulation.
- Binary tree expansion to an exact 32-leaf future set.
- Reverse collapse and probability cone shaping.
- Sliding-window dataset construction.
- Fused feature concatenation helpers.
- 36-to-100 feature expansion utilities for legacy checkpoint migration.
- Validation and sync scripts.

## Current Gaps After Refactor

Still blocked locally:
- `torch` is not installed in the local environment.
- `pandas`, `numpy`, and `pyarrow` are also not installed locally.
- Because of that, checkpoint loading and a real forward-pass validation remain blocked on the target server/runtime.

Current placeholder artifacts:
- Zero-filled local `news_embeddings.npy`, `news_embeddings_32.npy`, and `crowd_embeddings.npy` were generated so the fused 100-feature surface validates end-to-end.
- The local `news_emb_index.parquet` and `crowd_emb_index.parquet` files are placeholders and should be regenerated on the server with real parquet writers once the offline embedding pipelines run.

## Recommendations

Immediate next steps:
- Run the new numbered notebooks on the ROCm server instead of rebuilding logic inside notebooks.
- Train or regenerate the real 32-dim news and crowd embeddings into `data_store/embeddings/`.
- Load the existing TFT checkpoint on the server and validate the migration path with `src/models/nexus_tft.py`.
- Use `scripts/validate_pipeline.py` again once server dependencies and embedding files exist.

Repo hygiene improvements already made:
- Legacy notebook code was removed and replaced by thin wrappers.
- Shared config now drives paths and feature widths.
- ABM/MCTS/model utilities are now testable without notebook execution.
