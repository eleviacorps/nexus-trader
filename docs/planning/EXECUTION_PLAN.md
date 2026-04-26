# EXECUTION_PLAN

## Notebook Order

1. `00_environment_setup.ipynb`
2. `01_data_download.ipynb`
3. `02_price_pipeline.ipynb`
4. `03_news_pipeline.ipynb`
5. `04_crowd_pipeline.ipynb`
6. `05_persona_simulation.ipynb`
7. `06_feature_fusion.ipynb`
8. `07_tft_training.ipynb`
9. `08_future_branching.ipynb`
10. `09_reverse_collapse_and_ui.ipynb`
11. `10_validation_and_tests.ipynb`

## Canonical Cloud Layout

- `data/raw/price`
- `data/raw/news`
- `data/raw/crowd`
- `data/raw/macro`
- `data/processed`
- `data/embeddings`
- `data/features`
- `data/branches`
- `models/news_projection`
- `models/crowd_projection`
- `models/personas`
- `models/tft`
- `models/collapse`
- `outputs/charts`
- `outputs/probability_cones`
- `outputs/evaluation`
- `outputs/logs`

## Notes

- The repo stays aligned with the cloud-first structure while still supporting local fallback artifacts.
- The target architecture remains `World -> Perception -> Simulation -> Brain -> Future Branching -> Reverse Collapse -> Probability Cone UI`.
- The environment setup should install fast download tooling so cloud GPU time is spent on compute, not waiting on slow transfers.
- `scripts/download_core_datasets.py` is the first manifest-driven download entrypoint for macro, news, and crowd context collection.
- `scripts/validate_pipeline.py` now resolves canonical paths first and falls back to the current local artifacts when needed.
