# TODO_NEXT

## Server Execution

- Run `00_environment_setup.ipynb` on the ROCm server to create the canonical `data/`, `models/`, and `outputs/` tree.
- Install `aria2`, `pigz`, and `p7zip-full` on the server before any large direct-file downloads.
- Run `python scripts/download_core_datasets.py --plan` and then execute category downloads in priority order.
- Move from placeholder local artifacts to real server-generated parquet and tensor outputs.
- Re-run `10_validation_and_tests.ipynb` after each major artifact stage.

## News And Crowd Perception

- Replace the local placeholder embedding artifacts with real `news_embedding.parquet` and `crowd_embedding.parquet` outputs sourced from downloaded raw datasets.
- Regenerate the embedding index parquet files with a real parquet writer on the server.
- Add metric reports for semantic coverage, forward-fill coverage, and timestamp sparsity.

## Theory Alignment

- Use `THEORY_ALIGNMENT.md` as the north-star document for keeping the implementation faithful to the SVGs and project planning.
- Keep the LLM role constrained to interpretable macro-reading, not direct prediction.
- Expand personas toward ICT, SMC, Wyckoff, and macro-bias behaviors after the raw data layer is stable.

## Fused Training

- Validate the legacy checkpoint migration against a real ROCm `torch` runtime.
- Save `models/tft/final_tft.ckpt` and `outputs/evaluation/tft_metrics.json` after fused training.
- Add tensor artifact generation for `data/features/fused_tensor.npy` and `data/features/targets.npy`.

## Branching And UI

- Persist `data/branches/future_branches.json` from Notebook 08.
- Render `outputs/charts/probability_cone.html` and `outputs/charts/persona_breakdown.html` from Notebook 09.
- Promote the placeholder UI notebook into a real chart surface once the upstream artifacts exist.
