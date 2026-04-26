**Purpose**
This is the working execution plan for the remaining Nexus Trader build.

The goal is not to manufacture a fake `90%+` market-direction claim. The goal is to maximize real predictive quality, calibration, interpretability, and deployment readiness while keeping the system leakage-safe and operationally honest.

If a future run ever reports `90%+` short-horizon directional accuracy, that result must be treated as suspicious until it survives:
- walk-forward evaluation
- regime holdouts
- leakage review
- thresholded trade filtering
- slippage-aware backtesting

**North Star**
Nexus Trader should become:
- an interpretable multi-stream forecasting system
- a persona-driven branching simulator
- a reverse-collapse probability-cone engine
- a deployable service with MCP/tooling compatibility
- a chart-first UI that overlays price, cone, branch disagreement, and persona context

**Current Reality**
The repo already has:
- real macro/news/crowd ingestion from accessible sources
- aligned perception artifacts
- persona outputs and adaptive weights
- fused training artifacts
- a trainable TFT-like model path
- branching/collapse primitives
- remote ROCm execution history
- validation and tests

The repo does not yet have:
- a verified high-edge predictive model
- strong walk-forward backtesting
- robust regime-aware target engineering
- live branch scoring tied tightly to historical-fit logic
- a complete final UI workflow
- an MCP server surface

**Truthful Quality Goal**
Instead of chasing raw row-level classification accuracy, optimize for:
- directional precision on tradable high-conviction subsets
- calibration quality
- regime robustness
- stability across walk-forward periods
- explainable branch disagreement
- economic usefulness after thresholding and filtering

This matters because a trading system can be useful with modest global accuracy if:
- it filters aggressively
- it avoids low-consensus setups
- it sizes down when disagreement is high
- it only acts in aligned multi-signal conditions

**Remaining Workstreams**

1. Data Quality And Label Quality
- Rebuild targets from forward returns, not just legacy labels.
- Add volatility-aware thresholds so tiny noisy moves do not become training labels.
- Log target balance and horizon sensitivity across `1m`, `5m`, `15m`, and optional higher horizons.
- Preserve timestamp integrity and prevent future leakage in all forward-label generation.

2. Evaluation Integrity
- Move beyond random/ratio-like splits wherever possible.
- Use year-based walk-forward splits and explicitly log train/val/test years.
- Add evaluation summaries per regime:
  - low vol
  - high vol
  - trend
  - mean-reverting
  - event-heavy
- Treat “great” scores on one slice as invalid until repeated across multiple periods.

3. Model Capacity And Training Search
- Search over:
  - context window
  - hidden size
  - LSTM depth
  - dropout
  - learning rates
  - patience
  - threshold metric
- Compare pure real-label training vs simulation-supervised blended training.
- Store every experiment with config + metrics + timestamps.
- Favor calibration and walk-forward robustness over one-off peak accuracy.

4. Persona Logic Quality
- Tighten the retail/institutional/algo/whale/noise behavior rules.
- Push macro context harder into institutional and whale personas.
- Use crowd extremes as disagreement amplifiers, not just directional nudges.
- Track rolling persona usefulness over time and let weak personas lose influence.
- Expand dominant-driver reasoning so branch narratives remain interpretable.

5. Branch Quality
- Improve branch scoring so it reflects:
  - regime fit
  - persona alignment
  - macro/news/crowd coherence
  - path plausibility
  - volatility realism
- Preserve all leaves for reverse collapse.
- Expose both consensus and minority-risk scenarios in UI outputs.

6. UI And Product Surface
- Render real candlesticks plus the projected cone.
- Show:
  - current price
  - probability cone
  - consensus score
  - dominant driver
  - persona breakdown
  - branch summaries
- Keep the UI explainable rather than decorative.

7. MCP And Service Surface
- Expose metadata, prediction, latest cone, and latest market snapshot via MCP-compatible tools/resources.
- Keep FastAPI as the serving surface and add an MCP wrapper/server for agent integrations.

8. Publication And Persistence
- Keep the local repo authoritative.
- Push the local codebase to GitHub after changes are verified.
- Preserve durable handoff documents so the project can continue in future chats or new cloud instances.

**Concrete Execution Order**

1. Make the training pipeline configurable.
2. Add walk-forward/time-aware splitting.
3. Add richer target-generation hooks.
4. Improve branch scoring.
5. Build the final candlestick + cone UI generator.
6. Add MCP server support.
7. Run:
   - local tests
   - pipeline validation
   - remote smoke training
   - larger remote training sweeps when access is available
8. Publish verified repo state to GitHub.

**Success Criteria**
- The project remains fully reproducible.
- New configs and model runs are logged and explainable.
- The UI reflects real branch disagreement.
- MCP/service integration works from the codebase.
- Performance reporting is honest.
- No claims of `90%+` accuracy are made without strict evidence.

**Failure Modes To Avoid**
- leakage through labels or alignment
- overfitting a single date range
- reporting thresholded trade hit-rate as if it were overall model accuracy
- conflating backtest win-rate with raw prediction accuracy
- overcomplicating the architecture before evaluation is trustworthy

**What “Best Possible Outcome” Looks Like**
The most realistic strong outcome is:
- modest but real predictive edge
- stronger performance in selective high-consensus situations
- a useful probability-cone interface for manual or semi-automated decision support
- a robust experimentation pipeline that keeps improving over time

That is the path this repo should optimize for.
