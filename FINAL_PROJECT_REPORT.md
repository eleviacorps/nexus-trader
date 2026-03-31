**Final Project Report**

**What Nexus Trader Was Supposed To Be**

Nexus Trader was designed to be an interpretable, simulation-first market forecasting system.

The original vision was:
- ingest market reality
- separate price/news/crowd perception
- simulate different trader archetypes
- branch into multiple plausible futures
- collapse those futures back into a probability cone
- show both direction and disagreement in a final UI

It was never meant to be a plain indicator bot.

**What Existed Before This Session**

Before this session, the repo already had:
- a major refactor out of notebook-only prototypes
- real perception pipelines for macro/news/crowd
- persona outputs and simulation supervision artifacts
- a trainable TFT-like model path
- MCTS/reverse-collapse primitives
- a minimal inference/deployment surface
- a validated remote ROCm runtime on the Jupyter server

It was conceptually aligned with the intended architecture, but still incomplete in execution and still far from a proven high-accuracy model.

**What Was Added In This Session**

Planning and documentation:
- [MODEL_QUALITY_EXECUTION_PLAN.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/MODEL_QUALITY_EXECUTION_PLAN.md)
- [PROJECT_MASTER_SUMMARY.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/PROJECT_MASTER_SUMMARY.md)
- [MCP_READY.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/MCP_READY.md)
- [README.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/README.md)
- updates to [CONTEXT_HANDOFF.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/CONTEXT_HANDOFF.md)
- updates to [DEPLOYMENT.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/DEPLOYMENT.md)

Model/training upgrades:
- configurable training parameters in [train_fused_tft.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/train_fused_tft.py)
- sample-weighted training support in [train_tft.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/training/train_tft.py)
- year-based split helper in [training_splits.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/utils/training_splits.py)
- sample-weight artifact generation in [build_fused_artifacts.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/build_fused_artifacts.py)
- sequence dataset support for sample weights in [fused_dataset.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/data/fused_dataset.py)

Branching/product surface:
- richer branch-node paths in [tree.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/mcts/tree.py)
- candlestick/dashboard rendering in [render.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/ui/render.py)
- branching/dashboard generation in [build_branching_ui.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/scripts/build_branching_ui.py)
- MCP server surface in [server.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/mcp/server.py)
- expanded service surface in [app.py](C:/PersonalDrive/Programming/AiStudio/nexus-trader/src/service/app.py)

Notebook/UI flow:
- updated [08_future_branching.ipynb](C:/PersonalDrive/Programming/AiStudio/nexus-trader/notebooks/08_future_branching.ipynb)
- updated [09_reverse_collapse_and_ui.ipynb](C:/PersonalDrive/Programming/AiStudio/nexus-trader/notebooks/09_reverse_collapse_and_ui.ipynb)

**What Was Executed Remotely In This Session**

Cloud repo:
- pushed the project state to `nexus_trader_c`

Remote Jupyter runtime:
- resynced the latest local code to `/home/rocm-user/jupyter/nexus`
- rebuilt fused artifacts with sample weights
- confirmed remote tests pass
- confirmed remote validation passes
- ran a larger upgraded training pass
- generated the final branch/dashboard outputs
- ran a first direct model prediction from the trained checkpoint
- pulled the resulting checkpoint and reports back to the local repo

**Remote Results Pulled Back**

Training config:
- sequence length: `180`
- hidden dim: `192`
- layers: `2`
- dropout: `0.15`
- sample weighting: `enabled`
- simulation supervision: `enabled`
- sample run size: `1,000,000`

Metrics:
- validation accuracy: `0.5164`
- test accuracy: `0.5117`
- validation ROC-AUC: `0.5174`
- test ROC-AUC: `0.5142`

First direct prediction:
- bullish probability: `0.4834`
- bearish probability: `0.5166`
- signal: `bearish`

Dashboard snapshot:
- current price: `4493.448`
- branch count: `32`
- consensus score: `0.970468`
- dominant driver: `crowd_buying`

**Hard Truth**

This session did not produce a true `90%+` prediction model.

It produced:
- a stronger and more complete codebase
- a better training surface
- preserved cloud artifacts
- a working branch/candlestick dashboard
- an MCP-ready code path
- another honest remote training result

The actual predictive edge remains weak by raw classification metrics.

That is not failure. It is the truthful state of the project.

**What Is Finished Enough To Use**

You now have:
- a local repo with the latest pulled-back checkpoint and reports
- a candlestick dashboard artifact
- a probability cone artifact
- persona breakdown HTML
- an MCP server module
- a README and durable planning/summary docs
- two GitHub repos with code pushed during this broader workflow

**What Still Needs Work**

Highest-value next steps:
- run larger year-based/walk-forward training jobs
- improve label engineering beyond legacy target direction
- add real trade-filter evaluation, not just raw row-level metrics
- improve branch scoring realism
- ensure remote service dependencies fully match the local service surface
- backtest the cone/consensus filter as a selective signal rather than chasing overall accuracy

**Recommended Next Chat Starting Point**

Start from:
- [README.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/README.md)
- [MODEL_QUALITY_EXECUTION_PLAN.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/MODEL_QUALITY_EXECUTION_PLAN.md)
- [PROJECT_MASTER_SUMMARY.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/PROJECT_MASTER_SUMMARY.md)
- [CONTEXT_HANDOFF.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/CONTEXT_HANDOFF.md)

Then continue with:
1. walk-forward/year-split remote runs
2. improved target engineering
3. thresholded backtesting and cone-based trade filtering
4. service hardening on the remote box

**Bottom Line**

Nexus Trader is now significantly more complete, better documented, GitHub-backed, cloud-synced, dashboard-producing, and MCP-aware than it was before this session.

But the model still needs honest, disciplined quality work before it can be called a real trading edge.
