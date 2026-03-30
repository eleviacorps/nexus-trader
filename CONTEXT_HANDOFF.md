# Nexus Trader Context Handoff

## Purpose

This file is the recovery document for a fresh chat.
If conversation history is lost, paste or summarize this file to restore project context quickly.

It covers:

- The intended product vision
- What has already been built
- What was verified locally and on the cloud server
- What is still missing
- The exact current direction for the project

## Project Vision

Nexus Trader is not intended to be a simple indicator bot.
The target design is an interpretable market-simulation and forecasting system where:

- real-world market inputs are collected
- multiple trader personas react differently to the same inputs
- a branching future tree is generated
- all branches are reverse-collapsed back into a consensus
- the final output is a live probability cone, not just buy/sell

The intended high-level flow is:

```text
WORLD
  -> PERCEPTION
  -> SIMULATION
  -> BRAIN
  -> FUTURE BRANCHING
  -> REVERSE COLLAPSE
  -> LIVE PROBABILITY CONE
```

Two SVGs from the user define the target architecture and are conceptually aligned with this repo direction:

- `C:\Users\rfsga\Downloads\full_system_concept (1).svg`
- `C:\Users\rfsga\Downloads\reverse_branch_collapse.svg`

## Important Truth About Current Status

The repo is aligned with the architecture in theory, but it does not yet fully implement the complete system described by the SVGs and long-form design notes.

Current status is best described as:

- conceptually aligned
- partially implemented
- runnable in several important places
- not yet a full production trading system

## What The User Wants

The user wants:

- a deployable production-grade version of Nexus Trader
- a cloud-runnable notebook pipeline from `00` through `10`
- richer data ingestion
- real model training on the cloud server
- a probability-cone UI
- eventually a trustworthy high-accuracy trading workflow

The user also expressed a target of `90%+` or even `95%+` predictive accuracy.

Important guidance already given:

- do not promise `90%+` short-horizon market-direction accuracy
- treat such numbers as suspicious until proven through walk-forward testing and leakage checks
- optimize for honest, real performance rather than magical backtest numbers

## Current Canonical Notebook Pipeline

The repo has been aligned to this ordered notebook flow:

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

## Cloud Server Access Already Verified

The user provided a Jupyter server and token.
Access was successfully verified and remote code execution worked.

Remote server used:

- URL: `http://129.212.178.105`
- Auth: token-based Jupyter API access

Note:

- This credential should be treated as sensitive.
- If this file is shared outside the user’s own environment, remove the token first.

## Remote Environment Facts Already Verified

The remote runtime was successfully accessed and tested.

Verified facts:

- Jupyter REST API auth works
- websocket kernel execution works
- remote Python execution works
- remote sync into `/home/rocm-user/jupyter/nexus` works
- remote validation works
- remote test suite works
- remote training runs work

Observed remote runtime:

- Python `3.12.3`
- `torch 2.5.1+rocm6.2`
- GPU available
- GPU name reported: `AMD Instinct MI300X VF`

## Major Refactor Already Done

The repo was refactored away from notebook-only prototype code into reusable modules.

Key areas now exist in code form:

- configuration
- fusion pipeline
- persona simulation
- MCTS / branching
- reverse collapse
- probability cone shaping
- fused dataset loading
- training utilities
- inference API
- validation scripts
- tests

Important files added or upgraded include:

- [config/project_config.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/config/project_config.py)
- [src/pipeline/fusion.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/pipeline/fusion.py)
- [src/data/fused_dataset.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/data/fused_dataset.py)
- [src/simulation/personas.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/simulation/personas.py)
- [src/simulation/abm.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/simulation/abm.py)
- [src/mcts/tree.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/mcts/tree.py)
- [src/mcts/reverse_collapse.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/mcts/reverse_collapse.py)
- [src/mcts/cone.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/mcts/cone.py)
- [src/models/nexus_tft.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/models/nexus_tft.py)
- [src/training/train_tft.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/training/train_tft.py)
- [scripts/build_fused_artifacts.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/build_fused_artifacts.py)
- [scripts/train_fused_tft.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/train_fused_tft.py)
- [src/service/app.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/service/app.py)
- [scripts/validate_pipeline.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/validate_pipeline.py)

## Deployment Work Already Added

Basic deployment-facing assets now exist:

- [DEPLOYMENT.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/DEPLOYMENT.md)
- [Dockerfile](C:/PersonalDrive/Programming/AiStudio/nexus-trader/Dockerfile)
- `requirements-prod.txt`

The inference layer now supports:

- `/health`
- `/metadata`
- `/predict`

The inference API validates sequence shape and uses a saved manifest-backed threshold.

## Model/Training Enhancements Already Added

The training/export path was strengthened to produce:

- fused feature artifact
- targets artifact
- timestamps artifact
- model checkpoint
- model manifest
- evaluation metrics
- calibration report
- feature importance report
- training summary

Important output paths now include:

- `models/tft/final_tft.ckpt`
- `models/tft/model_manifest.json`
- `outputs/evaluation/training_summary.json`
- `outputs/evaluation/tft_metrics.json`
- `outputs/evaluation/calibration_report.json`
- `outputs/evaluation/feature_importance.json`

## What Was Verified Locally

Local verification completed successfully after the refactor and deployment work.

Successful local checks:

- `python -m unittest discover -s tests -p "test_*.py" -v`
- Python compile validation over the repo

Latest local test count at the time of this handoff:

- `18` tests passing

## What Was Verified Remotely

The following remote checks were successfully executed on the cloud server:

- `python scripts/validate_pipeline.py`
- `python -m unittest discover -s tests -p 'test_*.py' -v`
- `python scripts/build_fused_artifacts.py --limit-rows 200000`
- `python scripts/build_fused_artifacts.py`
- `python scripts/train_fused_tft.py --epochs 1 --batch-size 256 --sample-limit 200000`
- `python scripts/train_fused_tft.py --epochs 1 --batch-size 256 --sample-limit 1000000`
- later rerun after training/export improvements:
  - `python scripts/train_fused_tft.py --epochs 2 --batch-size 256 --sample-limit 1000000`

## Real Remote Data Facts Observed

Remote data inspection showed real artifacts already present on the server.

Observed remotely:

- price parquet rows: about `6,024,602`
- aligned news tensor shape: `(6024602, 32)`
- aligned crowd tensor shape: `(6024602, 32)`
- embedding index parquet files exist

This means the remote environment is materially stronger than the local placeholder environment.

## Current Real Model Performance

Important:

- the system is runnable
- the code is more deployable
- but the predictive performance is still far from the user’s aspirational `90%+`

Representative remote sample-run results were approximately:

- validation accuracy around `52.4%`
- test accuracy around `51.4%`
- ROC-AUC around `0.51`

Interpretation:

- these are realistic weak-signal market-prediction numbers
- they do not support any claim of a high-confidence production trading edge yet
- better data, targets, regime logic, and leakage-safe evaluation are still needed

## Architecture Match Against The User’s Long Design Note

### Aligned In Spirit

These parts are already aligned in direction:

- world inputs
- persona-based market simulation
- branching future tree
- reverse collapse
- probability cone concept
- interpretable components
- cloud-first pipeline

### Not Fully Implemented Yet

These parts from the user’s design are not fully present yet:

- live `GDELT + feeds` ingestion
- daily `MCP -> external LLM -> weekly macro thesis` loop
- frozen FinBERT `768` feature pipeline as specified
- separate `CNN + GRU` microstructure model
- true multi-horizon TFT predicting `5m / 1h / 4h / daily / weekly` simultaneously
- richer ICT/SMC/Wyckoff logic inside personas
- dynamic branch pruning/growth exactly matching the described `4 -> best 2 -> 4 more -> ... -> 64`
- full live chart that updates every `5` minutes on real feeds
- final manual-trading workflow with stable cone invalidation logic

## Existing Repo Notes

Other useful documents already in the repo:

- [PROJECT_AUDIT.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/PROJECT_AUDIT.md)
- [EXECUTION_PLAN.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/EXECUTION_PLAN.md)
- [TODO_NEXT.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/TODO_NEXT.md)
- [DEPLOYMENT.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/DEPLOYMENT.md)

## Recommended Next Direction

If resuming in a fresh chat, the best next move is not to promise unrealistic accuracy.
The best next move is to keep building the real system foundations.

Recommended next priorities:

1. Expand and improve datasets
2. Build cleaner news and crowd pipelines
3. Add macro/event alignment
4. Improve regime-aware labeling and backtesting
5. Add stronger persona logic
6. Add proper walk-forward evaluation
7. Build the live UI/cone outputs from real branch artifacts
8. Only then talk about execution or automation

## Data Expansion Guidance

The user asked whether we should download as much potentially useful data as possible.
Answer already given:

- yes, data expansion is valuable
- but not every extra dataset helps
- low-quality or leaky data can make the model worse

Best additions are likely:

- more clean OHLCV history across regimes
- macro time series aligned to bars
- timestamped news archives
- event calendars and surprise data
- filtered crowd/social datasets
- microstructure or tick-derived imbalance features where possible

## Current Honest Position To Tell Any New Chat

Use something close to this:

`The repo has already been refactored into reusable modules, synced to a working ROCm cloud server, validated remotely, and trained in smoke-run form on real data. It follows the user’s simulation -> branching -> reverse-collapse -> cone architecture in theory, but it does not yet fully implement the complete live system described in the SVGs. Current remote model performance is only around 51% to 52% on sample runs, so the project is not yet a high-accuracy production trading model. The next phase should focus on data quality, regime-safe evaluation, richer persona logic, and real live-perception pipelines rather than promising 90%+ accuracy.`

## Recovery Prompt For A New Chat

If needed, paste this into a new chat:

```text
We are working in the repo C:\PersonalDrive\Programming\AiStudio\nexus-trader.
Read CONTEXT_HANDOFF.md first and use it as the current project state.
This project is Nexus Trader, a cloud-first persona-simulation + branching + reverse-collapse + probability-cone trading research system.
The remote Jupyter ROCm server has already been accessed successfully, the repo has already been synced there, validation/tests have already passed remotely, and remote fused training smoke runs have already completed.
Do not assume the full SVG-described system is already implemented.
Treat the current system as partially implemented and currently only achieving about 51% to 52% sample-run accuracy remotely.
Continue from the next highest-value implementation step rather than rebuilding context from scratch.
```

## Final Reminder

This file is a high-signal summary, not a verbatim transcript of every message.
It is intended to preserve the important technical state and decision history.
