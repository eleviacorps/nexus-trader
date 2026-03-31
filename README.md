**Nexus Trader**

Nexus Trader is an interpretable market-simulation and forecasting project built around this core idea:

- markets are shaped by different participant types reacting differently to the same information
- short-horizon price movement is better represented as a distribution of plausible futures than a single deterministic line
- disagreement is signal, not noise to hide

The intended architecture is:

`WORLD -> PERCEPTION -> SIMULATION -> FUTURE BRANCHING -> REVERSE COLLAPSE -> PROBABILITY CONE`

**What The Repo Contains**

- perception pipelines for price, news, crowd, and macro context
- persona simulation outputs with rolling weight adaptation
- fused training artifacts for a TFT-like sequence model
- branch generation and reverse-collapse logic
- candlestick + probability-cone dashboard generation
- FastAPI inference surface
- MCP server surface
- notebook pipeline from `00` to `10`

**Current Honest Status**

This repo is materially closer to the intended product than the original notebook-only prototypes, but it is not yet a proven high-edge trading model.

The latest larger remote training pass pulled back into this repo used:

- `sequence_len = 180`
- `hidden_dim = 192`
- `lstm_layers = 2`
- `dropout = 0.15`
- simulation supervision: `enabled`
- sample weighting: `enabled`
- sample run size: `1,000,000`

Pulled-back metrics are currently around:

- validation accuracy: `0.5164`
- test accuracy: `0.5117`
- validation ROC-AUC: `0.5174`
- test ROC-AUC: `0.5142`

That is a believable weak-signal result, not a real `90%+` prediction system.

**Key Artifacts**

- model checkpoint: [final_tft.ckpt](C:/PersonalDrive/Programming/AiStudio/nexus-trader/models/tft/final_tft.ckpt)
- model manifest: [model_manifest.json](C:/PersonalDrive/Programming/AiStudio/nexus-trader/models/tft/model_manifest.json)
- training summary: [training_summary.json](C:/PersonalDrive/Programming/AiStudio/nexus-trader/outputs/evaluation/training_summary.json)
- metrics: [tft_metrics.json](C:/PersonalDrive/Programming/AiStudio/nexus-trader/outputs/evaluation/tft_metrics.json)
- latest snapshot: [latest_market_snapshot.json](C:/PersonalDrive/Programming/AiStudio/nexus-trader/outputs/evaluation/latest_market_snapshot.json)
- dashboard: [nexus_dashboard.html](C:/PersonalDrive/Programming/AiStudio/nexus-trader/outputs/charts/nexus_dashboard.html)

**First Pulled-Back Prediction**

Latest direct model inference from the trained checkpoint:

- bullish probability: `0.4834`
- bearish probability: `0.5166`
- signal: `bearish`
- threshold: `0.5050`
- sequence length: `180`

**Quick Start**

1. Build or refresh fused artifacts:

```bash
python scripts/build_fused_artifacts.py
```

2. Train:

```bash
python scripts/train_fused_tft.py --epochs 2 --batch-size 512 --sequence-len 180 --hidden-dim 192 --dropout 0.15 --sample-limit 1000000
```

3. Generate the final dashboard:

```bash
python scripts/build_branching_ui.py
```

4. Run the API:

```bash
pip install -r requirements-prod.txt
python -m src.service.app
```

5. Run the MCP surface:

```bash
python -m src.mcp.server
```

**Important Docs**

- [MODEL_QUALITY_EXECUTION_PLAN.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/MODEL_QUALITY_EXECUTION_PLAN.md)
- [PROJECT_MASTER_SUMMARY.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/PROJECT_MASTER_SUMMARY.md)
- [MCP_READY.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/MCP_READY.md)
- [CONTEXT_HANDOFF.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/CONTEXT_HANDOFF.md)
- [DEPLOYMENT.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/DEPLOYMENT.md)

**Next Best Work**

- stronger target engineering
- year-based and walk-forward training runs at scale
- more honest backtesting and thresholded trade evaluation
- better branch scoring tied to regime fit
- richer live service/API runtime dependencies on the server

This project should be treated as a serious, evolving research/deployment codebase, not as a finished alpha source.
