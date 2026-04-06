# Nexus Trader Master

## What This Project Is

Nexus Trader is a market simulator, not a normal trading bot.

The intended system is:

```text
WORLD
  -> PERCEPTION
  -> PERSONA SWARM
  -> FUTURE BRANCHING
  -> REVERSE COLLAPSE
  -> BRAIN / MODEL BIAS
  -> LIVE COMPARISON UI
```

The final output is supposed to be:

- a cloud of plausible short-term futures
- a reverse-collapsed consensus
- a confidence / disagreement measure
- a live comparison between predicted future and actual market behavior
- a human-readable technical context layer for manual analysis

The key design idea is that uncertainty should be visible, not hidden.

## What The User Wants

The user wants a system much closer to the MiroFish-style idea:

- multiple trader personas acting at once
- many branches of possible futures
- a reverse-confidence collapse back to the root
- a probability cone that reflects disagreement
- a live UI showing real candles beside predicted future candles
- manual observation first, not automatic execution
- technical-analysis helpers such as order blocks, structure, and key levels

The user also wants the project to be durable across chats and cloud resets.

## What Has Already Been Built

### Core Repo Structure

The repo has already been refactored into reusable modules and scripts.

Important implemented areas:

- `config/project_config.py`
- `src/pipeline/*`
- `src/simulation/*`
- `src/mcts/*`
- `src/models/*`
- `src/training/*`
- `src/service/*`
- `src/mcp/*`
- `src/ui/*`
- `scripts/*`
- `tests/*`

### Remote / Cloud Status

Remote Jupyter server access was already verified and used successfully.

Server:

- `http://129.212.178.105`
- token-based Jupyter API access

Remote runtime previously verified:

- Python `3.12.3`
- ROCm PyTorch available
- GPU reported as `AMD Instinct MI300X VF`

### Current Local Hosting

Local UI + API are now hosted through FastAPI / Uvicorn.

Usual local URLs:

- `http://127.0.0.1:8000/ui`
- `http://127.0.0.1:8000/api/simulate-live?symbol=XAUUSD`
- `http://127.0.0.1:8000/api/live-monitor?symbol=XAUUSD`
- `http://127.0.0.1:8000/api/llm/health`
- `http://127.0.0.1:8000/api/llm/context?symbol=XAUUSD`

### Local Artifacts Already Present

The essential cloud artifacts are already present locally:

- `models/tft/final_tft.ckpt`
- `models/tft/model_manifest.json`
- `outputs/evaluation/training_summary.json`
- `outputs/evaluation/tft_metrics.json`
- `outputs/evaluation/calibration_report.json`
- `outputs/evaluation/feature_importance.json`
- `outputs/charts/nexus_dashboard.html`
- `data/branches/future_branches.json`

## Current Honest State

This project is conceptually aligned with the intended architecture, but it is still only partially implemented.

Important truth:

- the codebase is much more complete than before
- the live UI exists
- the model can run locally
- branching and reverse-collapse primitives exist
- short-window historical analog scoring now exists for branches
- a quant-hybrid regime / volatility / fair-value layer now exists
- walk-forward evaluation and directional backtesting now exist as first-class tools
- GPT-OSS is now integrated as a local sidecar
- manual technical-analysis panels now exist
- but the full MiroFish-like simulation logic is still not complete

Also important:

- current predictive quality is still early
- previous remote sample runs were only around low-50% classifier performance
- first local walk-forward smoke evaluation still lands around ~50.8% calibrated directional accuracy on a capped 2024-2025 sample
- no honest claim of `90%+` predictive accuracy exists right now

## Latest Update

The completed cloud `v5` pass kept the stronger filtered-win-rate regime alive, and the project has now moved into a real `v6` redesign where the TFT is treated as the imagination engine and a new selector stack is being built to judge candidate futures.

Latest synced `v4` results:

- `mh12_full_v4`
  - strategic ROC-AUC: `0.5037`
  - `15m` ROC-AUC: `0.5097`
  - `30m` ROC-AUC: `0.5088`
  - trades: `153,504`
  - participation: `19.44%`
  - win rate: `64.70%`
- `mh12_recent_v4`
  - strategic ROC-AUC: `0.5068`
  - `15m` ROC-AUC: `0.4957`
  - `30m` ROC-AUC: `0.5057`
  - trades: `19,762`
  - participation: `25.60%`
  - win rate: `66.74%`

Important interpretation:

- the raw directional model is still only slightly better than random by ROC-AUC
- the filtered simulator stack is now materially better than the old low-50s regime on win rate
- the real edge is still coming from `selection + abstention + simulation structure`, not from a very strong standalone classifier

Latest synced `v5` results:

- `mh12_full_v5`
  - strategic ROC-AUC: `0.5082`
  - `15m` ROC-AUC: `0.5070`
  - `30m` ROC-AUC: `0.5097`
  - trades: `143,644`
  - participation: `18.20%`
  - win rate: `64.70%`
  - `$1000 fixed-risk`: `845,400`
  - max DD: `41.18%`
- `mh12_recent_v5`
  - strategic ROC-AUC: `0.5084`
  - `15m` ROC-AUC: `0.5059`
  - `30m` ROC-AUC: `0.5089`
  - trades: `24,240`
  - participation: `31.41%`
  - win rate: `65.70%`
  - `$1000 fixed-risk`: `153,240`
  - max DD: `13.55%`

Important `v5` interpretation:

- the internal regime router did not create a dramatic raw-model jump
- `30m` remains slightly stronger than `15m`
- the recent-regime `v5` run is still the cleaner filtered profile because drawdown is materially lower
- the strongest progress is still on `filtered win rate + participation control`, not raw standalone classifier strength

## Training Throughput Diagnosis

One important systems problem was finally identified clearly: the cloud box was not slow because the MI300X was weak. It was slow because the training / evaluation code was underfeeding it.

Main bottlenecks that were present:

- hardcoded `num_workers=0` in key training loaders
- hardcoded `num_workers=0` in walk-forward prediction loaders
- no AMP path in the main train/eval loop
- blocking host-to-device tensor transfers
- no configurable worker / prefetch / persistent-worker controls

What that caused:

- GPU compute often sat around `15% - 30%`
- CPU workers never ramped hard enough to keep batches flowing
- VRAM looked permanently high because ROCm / PyTorch was caching reserved memory even while compute was low

What has now been fixed locally:

- configurable worker counts in `config/project_config.py`
- AMP support in `src/training/train_tft.py`
- non-blocking tensor transfers in training and evaluation
- workerized loaders in `scripts/train_fused_tft.py`
- workerized walk-forward loaders in `src/evaluation/walkforward.py`
- remote `v6` runner now sets higher worker counts and AMP by default

Important honesty about the `90% VRAM` reading:

- on ROCm that does not necessarily mean the GPU is actively training
- it often means the allocator is holding reserved memory for the process
- low GPU utilization plus high VRAM reservation usually means the data pipeline or evaluation stage is the bottleneck, not that the card is “fully busy”

## Current Local Architecture Upgrade

After `v4`, the local code has now been pushed one step further toward a true routed predictor instead of only smarter filtering.

New local upgrades after the synced `v4` run:

- the TFT predictor now has an internal regime router plus expert heads
- inference now exposes routed-regime diagnostics
- ensemble weighting is now regime-aware instead of mostly static
- specialist bots now emit structural style-bias and regime-affinity summaries
- those bot regime/style summaries now feed directly into branch scoring and branch probability weighting
- live simulation rows now carry bot-regime context, not just a single aggregate swarm bias

This is the start of the intended `v5` direction:

```text
Quant / Regime State
  -> Routed Predictor Heads
  -> Specialist Bot Style Bias
  -> Branch Scoring + Collapse
  -> Regime-Aware Ensemble
```

GPT-OSS remains the sidecar / judge layer after these numeric builds. It is still not the raw forecaster.

## Current Remote `v5` Status

The routed-`v5` cloud cycle is now completed and synced locally.

Remote execution details:

- Jupyter workspace root: `nexus/`
- remote launcher script: `scripts/remote_v5_train.py`
- remote pid file: `outputs/logs/remote_v5_pipeline.pid`
- remote log: `outputs/logs/remote_v5_pipeline.log`
- final remote tail ended with: `===== v5 pipeline complete =====`

Completed pipeline shape:

1. `build_quant_context`
2. `build_persona_outputs`
3. `build_fused_artifacts`
4. remote tests
5. `train_mh12_full_v5`
6. `walkforward_mh12_full_v5`
7. `train_mh12_recent_v5`
8. `walkforward_mh12_recent_v5`

What `v5` proved:

- the new internal regime router is compatible with cloud retraining, walk-forward evaluation, and filtered backtesting
- regime-aware ensemble weighting preserved the stronger filtered win-rate regime
- deeper specialist-bot integration into branch scoring did not break branch generation or the live simulator path
- raw `15m/30m` ranking quality improved only marginally, so the next gains are still more likely to come from better targets, routing, and branch realism than from model size alone

## Current Local `v6` Architecture Direction

The project is now moving toward this structure:

```text
Current Market State
  -> Regime Detector
  -> Volatility / Range Envelope
  -> Historical Retrieval Prior
  -> TFT / Routed Predictor Generates Candidate Futures
  -> Branch Feature Extractor
  -> Branch Selector / Judge
  -> Reverse Collapse + Final Most-Realistic Future Path
```

This is different from the older assumption that the TFT itself should become the final predictor.

Current `v6` local modules now in the repo:

- `src/v6/regime_detection.py`
- `src/v6/volatility_constraints.py`
- `src/v6/historical_retrieval.py`
- `src/v6/branch_features.py`
- `src/v6/branch_selector.py`

What these modules do now:

- detect a higher-level market regime without predicting price directly
- estimate realistic `5m / 15m / 30m` movement envelopes
- retrieve historically similar states as an additional branch prior
- compute branch-level realism / plausibility features
- score branches with a dedicated selector model or a fallback learned-style score

This means the architecture is finally starting to reflect the user’s intended design:

- TFT = imagination engine
- selector = future judge
- collapse = uncertainty-preserving compression layer

## Current Remote `v6` Status

The cloud `v6` cycle has now been launched on the Jupyter ROCm box.

Remote execution details:

- Jupyter workspace root: `nexus/`
- remote launcher script: `scripts/remote_v6_train.py`
- remote pid file: `outputs/logs/remote_v6_pipeline.pid`
- remote log: `outputs/logs/remote_v6_pipeline.log`
- latest confirmed stage in this chat: `===== build_quant_context =====`
- latest confirmed remote pid: `751202`

Current `v6` cloud goals:

1. rebuild quant context with the faster loader path
2. rebuild persona outputs
3. rebuild fused artifacts
4. run tests
5. train `mh12_full_v6`
6. walk-forward evaluate `mh12_full_v6`
7. train `mh12_recent_v6`
8. walk-forward evaluate `mh12_recent_v6`

Main `v6` intent:

- keep `15m / 30m` as the primary product story
- make the generator faster to train and evaluate
- prepare the codebase for a real learned branch-selector layer instead of relying mostly on post-hoc gating

## What Was Fixed Recently

### Live Price Feed

XAUUSD was originally using `GC=F` futures directly, which caused a mismatch versus spot-style gold pricing.

Current approach:

- XAU uses `GC=F` chart candles
- then calibrates to a public spot gold quote from `gold-api`
- current market source is tagged as:
  - `spot_calibrated_gold_api_plus_gcf`

### UI

The live UI was upgraded to:

- dark theme
- live market chart
- predicted vs actual chart
- timestamped simulations
- auto-refresh
- history list
- cone hit-rate view
- top branch and minority branch overlays
- branch conversation panel
- manual technical-analysis panels

### Simulation Scaling Bug

There was a serious bug where branch probabilities were converted into prices using an unrealistic shortcut, causing absurd moves like gold jumping toward `4900` in 5 minutes.

That bug has been fixed.

Current cone logic now:

- aggregates actual branch price paths
- uses weighted mean branch prices
- uses weighted dispersion for cone width
- bounds confidence more realistically
- versions simulation history so stale broken runs are excluded

### Branching And Sidecar Integration

Recent branching improvements:

- each branch now rolls forward its own synthetic row state
- branch ranking includes:
  - probability weight
  - branch fitness
  - historical analog confidence
  - minority guardrail score
- the UI now shows:
  - strongest surviving branch
  - minority invalidation branch
  - supporting branches
- GPT-OSS now acts as a bounded sidecar:
  - it tilts personas
  - it contributes a small numeric prior
  - it does not replace the simulator

### Quant-Hybrid Layer

The project now includes a quant-finance support layer rather than relying only on the simulator plus TFT.

Current quant context includes:

- regime clustering / regime-strength estimation
- transition-risk estimation
- volatility realism scoring
- fair-value z-score estimation
- trend-score estimation

These quant features are now folded into:

- fused-target construction
- hold-mask / abstention logic
- sample weighting
- live simulation context
- branch scoring

Current local quant build on the synced slice:

- rows: `7611`
- feature columns: `14`
- average transition risk: about `0.0559`
- average volatility realism: about `0.8284`
- average regime strength: `1.0`

Current cloud quant rebuild on the full dataset:

- rows: `6,024,602`
- feature columns: `14`
- average transition risk: about `0.1271`
- average volatility realism: about `0.8096`
- average regime strength: about `0.8721`

This is still a support layer, not a standalone quant strategy engine.

### Specialist Bot Swarm

The live simulator now also includes a visible specialist bot layer.

Current design:

- 10 deterministic strategy bots
- each bot emits:
  - direction
  - confidence
  - key level
  - invalidation
  - 5m / 10m / 15m projected levels
  - short rationale
- current bot set includes:
  - trend bot
  - breakout bot
  - mean reversion bot
  - order block bot
  - liquidity sweep bot
  - fair value gap bot
  - macro regime bot
  - news shock bot
  - crowd extremes bot
  - risk skeptic bot

These bots are not the master predictor by themselves.

They feed into:

- the numeric ensemble
- persona reactions
- GPT swarm judgment
- UI graph views

### Analog Scoring, Walk-Forward Evaluation, And Backtesting

The repo now has a first real evaluation layer instead of only live UI inspection.

New pieces:

- `src/mcts/analog.py`
- `src/evaluation/walkforward.py`
- `scripts/run_walkforward_evaluation.py`

What analog scoring does now:

- samples a historical memory bank from fused features
- compares the live / simulated state against short windows of similar past price-structure regimes
- uses both:
  - the latest row state
  - the recent multi-bar regime path
- returns:
  - bullish probability
  - directional bias
  - analog confidence
  - support count
- injects that into branch ranking, reverse-collapse weighting, and live branch expansion

What walk-forward evaluation does:

- loads the current trained TFT checkpoint
- evaluates by year using timestamped fused artifacts
- applies bucket calibration from validation years
- reports raw metrics, calibrated metrics, calibration curves, and a simple directional backtest

Important honesty:

- the current backtest is a directional unit-PnL backtest, not a broker-grade execution backtest
- it is meant to measure selective correctness and confidence behavior first
- it is not yet a slippage/commission/execution simulator

Latest local smoke run:

- years: `2024, 2025`
- capped sample: `500` windows per year
- calibration years: `2021, 2022, 2023`
- overall calibrated directional accuracy: about `50.8%`
- current edge is still weak

That is not a failure of this phase.
It is the baseline we need before deeper branch logic, regime routing, and higher-quality targets can be judged honestly.

What changed after that smoke run:

- analog matching was upgraded from single-row similarity to short-window regime analogs
- live branching now passes recent price-feature history into the analog scorer
- branch analog fitness is now based on evolving regime context, not only a point snapshot

### GPT Judge Layer

GPT-OSS now has a second structured role beyond market context:

- judge the 10-bot swarm
- compare the bot swarm to the simulator and branch outputs
- read raw headline feed and public-discussion feed during judgment
- read technical context including order blocks and fair value gaps
- read a 120-bar chart snapshot summary
- summarize the debate
- identify strongest and weakest specialist
- describe the minority case
- produce a final manual stance:
  - buy
  - sell
  - hold
- keep the result as structured JSON

This is still a sidecar layer, not a replacement for the simulator.

### Manual Technical Analysis Layer

The live UI now includes a manual-analysis layer with:

- structure / session read
- RSI and ATR context
- equilibrium and premium-discount context
- nearest support / resistance
- order block detection
- fair value gap detection
- order-block overlays on the live chart

This matters because the simulator should help a human read the market, not only emit a cone.

Important note on external order-block APIs:

- no trustworthy free public order-block API has been adopted
- order blocks are still computed locally from price structure
- this is currently preferred over adding a fragile external dependency

### UI Terminal Refresh

The UI is now being reshaped toward a cleaner financial-terminal layout instead of a long sidebar stack.

Current terminal direction:

- clean live tape chart
- predicted-vs-actual multi-horizon chart
- final GPT judge panel
- specialist bot board
- public reaction theater
- branch graph view
- swarm graph
- TradingView desk panel

The goal is to keep:

- one area for live market
- one area for the simulator forecast
- one area for final judgment
- one area for raw feeds
- one area for graphs and manual chart confirmation

### Multi-Horizon Forecasting

The live simulator is moving away from a 5-minute-only view.

Current direction:

- selectable forecast horizons
  - 5m
  - 10m
  - 15m
  - 30m
- one combined final forecast path built from:
  - branch simulator
  - bot swarm aggregate
  - GPT sidecar tilt
  - base model bias

This makes the compare chart closer to a final consensus path rather than only the raw simulator center.

### Model Routing

The LLM sidecar path is no longer assumed to be one backend only.

Current direction:

- selectable provider routing
  - `LM Studio Local`
  - `Ollama Cloud`
- both market-context extraction and swarm judgment can be routed through the selected provider
- the app still treats the LLM as a sidecar, not the main numeric predictor

## What Is Still Missing Relative To The Real Vision

These pieces are still underbuilt compared with the target architecture:

- deeper persona behavior
- real regime-dependent branch scoring
- better reverse-collapse quality logic
- historical analog matching
- branch pruning / regrowth logic closer to the MiroFish idea
- stronger confidence calibration
- better use of macro/news/crowd in the actual branch simulator
- walk-forward evaluation and reliability tracking
- deeper smart-money-style structural logic
- stronger technical overlays and annotation quality
- historical validation of bot-swarm usefulness
- better multi-horizon consistency checks across bots and branches
- cleaner information hierarchy in the UI

## What “MiroFish Functionality” Should Mean Here

To get closer to the intended design, the branching system should behave like this:

1. Build one current market state from price + macro + news + crowd.
2. Spawn many variant futures from that same state.
3. Each branch should differ because of:
   - persona composition
   - regime assumptions
   - shock/noise assumptions
   - micro-structure path assumptions
4. Each branch should simulate multiple future candles, not a single probability.
5. Each branch should get scored by:
   - internal plausibility
   - consistency with current regime
   - historical analog fit
   - agreement / disagreement with the neural model bias
   - structural validity
6. Reverse collapse should preserve every leaf as a vote.
7. Final confidence should come from:
   - branch directional agreement
   - branch price dispersion
   - branch fit quality
   - model calibration
   - realism of the projected move

That is much closer to the real target than the current lighter implementation.

## What To Improve Next

### 1. Branching Engine

Highest-value work:

- branch over simulated candle paths, not just repeated ABM steps
- store branch state per timestep
- compute branch fitness using historical analog windows
- keep more than one “winning” path alive
- collapse from all leaves, never a single winner
- keep an explicit minority scenario even in strong-consensus regimes
- connect branch fitness more directly to the specialist bot swarm

### 2. Reverse Collapse

Current collapse is still too simple.

Improve it by adding:

- subtree confidence
- branch fitness score
- regime consistency score
- model agreement score
- explicit minority-scenario tracking

Then output:

- consensus path
- cone width
- minority risk scenario
- confidence score

### 3. Confidence Logic

Current confidence still feels too random / too high too often.

Make confidence depend on:

- branch dispersion
- branch directional agreement
- branch fit to current regime
- ATR-relative move credibility
- historical reliability of similar branches

Never let confidence become high only because all branches happen to agree on an unrealistic move.

### 4. Persona Logic

Current personas are still heuristic and shallow.

Need to deepen:

- retail trend chasing / crowd following
- institutional macro and value logic
- whale contrarian accumulation / distribution logic
- algo short-term structure / imbalance logic
- noise randomness with bounded influence

Also add regime-specific behavior:

- calm range
- trend
- macro shock
- panic / liquidation
- post-news fade

### 5. Technical Market Structure

The simulator should get better at reading structure the same way a discretionary human would.

Important additions:

- higher-timeframe dealing range context
- better order-block validation
- liquidity pool detection above highs / below lows
- fair value gap prioritization
- premium / discount logic
- stronger swing failure pattern recognition

These should inform both:

- branch scoring
- UI overlays for manual reading

### 6. Evaluation

Current project still needs stronger evaluation than plain classifier metrics.

Priorities:

- walk-forward testing by year/month
- selective signal precision
- simulation hit-rate
- cone containment rate
- directional hit-rate after filtering
- regime-by-regime performance
- minority-scenario rescue rate
- bot-vs-simulator agreement calibration
- judge stability under repeated runs

### 7. UI / Monitoring

Need to keep improving:

- compare chart
- branch explorer
- minority scenario overlay
- branch-by-branch explanation
- source timestamps for every feed
- stale-feed warnings
- richer technical overlays
- better terminal-style layout and panel hierarchy
- optional TradingView/manual chart workspace

## Should We Use A Local LLM?

Short answer:

- yes, as a sidecar
- no, not as the main numeric predictor

### Good Uses For A Local LLM

A local LLM can help with:

- macro headline summarization
- event severity classification
- daily / weekly macro thesis generation
- crowd narrative summarization
- branch explanation in plain language
- offline labeling and regime tagging
- extracting structured facts from messy text

That is where an LLM is useful.

### Bad Uses For A Local LLM

A local LLM should not be the main next-5-minute price predictor.

Why:

- LLMs are weak at raw numeric time-series forecasting
- they are expensive for high-frequency inference loops
- they can hallucinate structure
- they are harder to calibrate than specialized sequence models

So the main predictor should still be:

- time-series model
- branch simulator
- reverse-collapse engine

The LLM should be an interpretability and perception assistant, not the market oracle.

## On Building Trading Bots

The cleanest stance is:

- Nexus Trader should remain a market simulator first
- execution bots, if added, should sit downstream as optional consumers

The idea of running many bots in parallel is not wrong, but it will not magically create `90%+` accuracy.

Better version of that idea:

- keep one core simulator
- add multiple execution-policy agents downstream
- examples:
  - trend continuation policy
  - reversal policy
  - breakout policy
  - mean-reversion policy
  - no-trade filter policy
- human overseer approves or rejects

That can improve decision quality because different policies specialize in different regimes.

What should not happen:

- 10 random bots all placing equal-weight opinions
- treating vote count as truth

If multi-agent execution is ever added, it should be:

- simulator first
- regime filter second
- strategy policies third
- human oversight on top

### Better Interpretation Of “10 Bots”

The useful version is not 10 random bots voting blindly.

The useful version is:

- one core simulator
- one model layer
- one local LLM judge
- 10 specialist strategy bots that expose interpretable views

So the 10 bots should be treated as:

- visible specialist analysts
- structured policy heads
- manual-analysis helpers

not as autonomous execution engines.

## On LM Studio Parallelism

For the current Nexus use case, GPT-OSS is a sidecar, not the main engine.

That means LM Studio parallelism should be chosen for:

- low latency
- stable memory usage
- reliable JSON output

Practical recommendation:

- `4` parallel is reasonable right now
- increase only if:
  - GPU memory headroom is clearly available
  - latency does not spike
  - JSON reliability does not worsen

For sidecar interpretation calls, higher parallelism is often worse if it causes:

- context swapping
- slower first-token time
- unstable response timing

So the default recommendation is:

- keep `4` for now
- only test `6` or `8` if the machine remains smooth
- do not increase it just because more sounds better

With an RTX 4070 and 16GB RAM:

- `4` is still the safest default for GPT-OSS sidecar usage
- `6` may be worth testing only if latency stays acceptable
- `8+` is likely to hurt responsiveness more than it helps for this project

## Recommendation On Specific Local Models

Based on current official/public model info:

- OpenAI `gpt-oss-20b` is designed for local inference and strong tool use, and OpenAI says it is suited for agentic workflows and can run with relatively modest memory for its class.
  Source: OpenAI official announcement
  Link: https://openai.com/index/introducing-gpt-oss/

- Qwen2.5 is available across many sizes, and Qwen’s official model card shows strong instruction, math, and coding performance in its family.
  Source: Qwen official model card
  Link: https://qwen2.org/qwen2-5/

- Meta Llama 3.1 8B Instruct is self-hostable and supports tool calling / customization in NVIDIA’s hosted system card view.
  Source: NVIDIA model/system card for Meta Llama 3.1 8B
  Link: https://build.nvidia.com/meta/llama-3_1-8b-instruct/systemcard

### My Practical Recommendation

For this project:

- use `gpt-oss-20b` only if you want a stronger local reasoning/tool-use sidecar and have enough memory budget
- use `Qwen2.5-7B/14B Instruct` if you want a lighter structured-text / summarization model
- use `Llama 3.1 8B Instruct` only if your local deployment stack already fits it well

Best architecture choice:

- keep the numeric predictor separate
- use one local LLM as a tool-calling perception / explanation agent
- optionally fine-tune a much smaller classifier on your own domain labels instead of trying to make the LLM itself forecast price

## Best Next Technical Direction

If continuing seriously, the best next build order is:

1. Add an explicit regime classifier and route branch scoring by regime.
2. Retrain on the cloud with the richer analog/regime signal folded into scoring and supervision.
3. Improve reverse collapse and minority scenario logic further.
4. Improve confidence calibration from actual branch reliability.
5. Add a precision gate / no-trade gate on top of the numeric stack.
6. Keep GPT-OSS as a structured sidecar.
7. Use uncapped walk-forward reports and stricter filtered backtests after every major scoring change.
8. Only then consider more advanced training loops.

## Suggested LLM Role In This Project

Use a local or self-hosted LLM like this:

### Input

- latest macro headlines
- latest crowd/news items
- regime features
- branch summaries

### Output

Return strict JSON fields like:

- `macro_thesis`
- `event_severity`
- `dominant_narrative`
- `risk_of_regime_shift`
- `institutional_bias`
- `whale_bias`
- `explanation_for_top_branch`

### Rules

- no free-form trading advice
- no direct buy/sell decision
- no direct numeric price target
- only structured interpretation of text and regime context

### Fine-Tuning Recommendation

Do not fine-tune GPT-OSS first for short-horizon price prediction.

Why:

- the current weakness is still mostly in labels, regime handling, branch realism, and confidence calibration
- a fine-tuned LLM will not fix weak numeric supervision or noisy short-term market labels
- it is very easy to build an expensive narrative model that sounds smarter while the actual walk-forward win rate barely moves

If fine-tuning is used at all, the best order is:

1. fine-tune or train a small regime / event classifier first
2. fine-tune a smaller text model for:
   - macro regime labeling
   - event severity
   - crowd narrative polarity
   - branch explanation
3. only after the numeric simulator improves should any LLM fine-tuning be considered for structured sidecar tasks

Best practical recommendation right now:

- keep GPT-OSS as a sidecar judge and structured text interpreter
- improve the numeric simulator first
- if a model is trained next, prioritize:
  - regime classifier
  - meta-label precision gate
  - small text/event classifier
  - cloud retraining of the numeric model with richer regime analog features

## Latest Cloud Run

A fresh remote ROCm run was completed again on the Jupyter server after reconnecting to `129.212.178.105`.

What was done remotely:

- verified GPU/runtime and artifact availability
- rebuilt `news_embeddings.npy` and `crowd_embeddings.npy` from raw cloud datasets
- confirmed those aligned tensors are sparse in the early history but non-zero in the recent tail
- rebuilt `data/features/fused_features.npy` from the refreshed embeddings
- ran full year-split TFT retraining on the cloud GPU
- ran walk-forward evaluation for `2024, 2025, 2026`
- pulled the updated checkpoint and evaluation reports back to local

Important remote findings:

- the aligned recent-tail embeddings are real and non-zero
- the full-history aligned embeddings are still sparse because the no-auth news/crowd sources are mostly recent
- this means cross-modal signal exists, but it is not uniformly strong across the full 2009 -> 2026 training span

Latest remote training result:

- validation ROC-AUC: about `0.5114`
- test ROC-AUC: about `0.5101`
- the learned threshold optimized for F1 collapsed to `0.05`, which is a warning sign that separation is still weak

Latest remote walk-forward result:

- overall raw accuracy: about `50.76%`
- overall ROC-AUC: about `0.5101`
- filtered backtest participation: about `20.67%`
- filtered backtest win rate: about `50.32%`
- 2026 fold win rate: about `51.77%`

Interpretation:

- the system remains weak but slightly positive in the best filtered recent slice
- this is still not remotely close to a trustworthy `90%+` engine
- the next gains are still much more likely to come from target redesign, alignment fixes, regime routing, and stronger filtering than from bigger models alone

## Latest Multi-Horizon Retrain

Another remote MI300X pass was completed after the target redesign and model upgrade.

What changed in that run:

- fused artifacts were rebuilt with the new hold-aware multi-horizon targets
- the TFT head was upgraded to emit:
  - `5m`
  - `10m`
  - `15m`
  - `30m`
- a lightweight precision gate was trained from validation behavior
- two separate remote training runs were executed:
  - full-history model
  - recent-regime model

### Full-History Multi-Horizon Result

Split:

- train: `2009 -> 2020`
- val: `2021 -> 2023`
- test: `2024 -> 2026`

Key results:

- `5m` test ROC-AUC: about `0.5107`
- `10m` test ROC-AUC: about `0.5145`
- `15m` test ROC-AUC: about `0.5145`
- `30m` test ROC-AUC: about `0.5143`

Interpretation:

- the multi-horizon head is slightly more useful away from the shortest horizon
- the primary `5m` edge is still very weak
- the precision gate collapsed in this run and filtered everything out, which means the current gate feature set is not yet strong enough

### Recent-Regime Model Result

Split:

- train: `2021 -> 2024`
- val: `2025`
- test: `2026`

Key results on `2026`:

- `5m` test ROC-AUC: about `0.5084`
- `10m` test ROC-AUC: about `0.5139`
- `15m` test ROC-AUC: about `0.5160`
- `30m` test ROC-AUC: about `0.5250`

Interpretation:

- the recent-regime model still did not beat the `5m` ceiling meaningfully
- but the longer horizons improved more than the primary horizon
- that is a useful sign: the system appears to learn better at slightly slower horizons than at ultra-short directional timing
- the recent-model gate became too permissive and passed everything, so it also needs another iteration

### Honest Conclusion From This Pass

This pass improved the architecture more than it improved the tradable edge.

What got better:

- cleaner supervision
- real multi-horizon outputs
- separate recent-regime path
- proper gate artifact generation
- better evaluation clarity

What did not improve enough:

- `5m` primary edge
- abstain quality
- selective win rate

The most important new lesson is:

- the project may have more signal at `15m / 30m` than at raw `5m`
- if we keep forcing the evaluation story to be mostly about `5m`, we may be throwing away the part of the model that is actually learning something

## Latest Hold / Confidence Upgrade

The next architectural pass moved beyond plain directional horizon outputs.

What changed:

- each horizon now has three explicit supervised surfaces:
  - `direction`
  - `hold`
  - `confidence`
- the target builder now writes these into `targets_multihorizon.npz`
- the model head can now emit `12` outputs:
  - `target_5m`
  - `target_10m`
  - `target_15m`
  - `target_30m`
  - `hold_5m`
  - `hold_10m`
  - `hold_15m`
  - `hold_30m`
  - `confidence_5m`
  - `confidence_10m`
  - `confidence_15m`
  - `confidence_30m`

Why this matters:

- abstention can now be modeled directly instead of bolted on later
- the gate can reason over:
  - horizon disagreement
  - horizon hold likelihood
  - horizon confidence quality
- evaluation can now treat `15m / 30m` as a strategic surface instead of an afterthought

Important practical note:

- the local smoke run confirms the new head and reports execute end-to-end
- this is still an early structural upgrade, not proof of a better live edge yet

## Latest Strategic Update

The analog layer has now been upgraded from single-row similarity toward short-window regime analogs.

What changed:

- branch scoring can now carry a rolling short-window analog history
- live sequence construction now computes analog context from a recent feature window instead of a single latest row
- the analog report now describes regime mode and window size
- the fused-target pipeline no longer relies on legacy `target_direction` as the main supervision source
- targets are now rebuilt from forward returns over `5m / 10m / 15m / 30m`
- the primary training label stays directional for compatibility with the current binary model, but low-signal hold windows are now explicitly down-weighted instead of being treated as equally important
- sample weights now reward:
  - ATR-aware move significance
  - multi-horizon agreement
  - stronger forward moves
- new local artifacts now include:
  - `data/features/target_hold_mask.npy`
  - `data/features/targets_multihorizon.npz`
- the current local target distribution after the redesign is much healthier than the first draft:
  - primary 5m hold rate around `24.4%`
  - sample-weight mean around `2.14`
  - multi-horizon agreement around `0.77`

What is still blocked:

- cloud retraining has not been run in this chat because the current environment does not have live access to the remote Jupyter box
- uncapped walk-forward evaluation after retraining is therefore still pending

## Highest-Impact Win Rate Blockers

A focused codebase review surfaced the main reasons win rate is still weak. These matter more than model size right now.

1. The training target is still too close to noisy next-bar direction instead of the true multi-horizon simulator objective.
2. News and crowd modalities still need to be treated as suspect until we fully confirm they are non-zero and timestamp-aligned in the training artifacts.
3. Fusion still needs stricter timestamp alignment guarantees instead of relying on row-count compatibility.
4. Evaluation is improving, but true rolling retrain walk-forward is still not the default path.
5. The current TFT surface is still closer to a binary sequence classifier than a full multi-horizon market simulator head.

Important update:

- blocker `#1` has now been partially reduced by the new target redesign
- the repo still needs cloud retraining and new evaluation before we can claim the redesign improved real win rate
- blocker `#5` remains real, because the model output head is still binary even though the supervision artifacts are now multi-horizon-aware

## LLM Fine-Tuning Recommendation

Current recommendation: do not LoRA-fine-tune GPT-OSS yet for numeric price prediction.

Why:

- the biggest bottlenecks are target definition, modality quality, and alignment
- a larger or fine-tuned LLM will not fix noisy labels or dead features
- LLMs are still better here as structured perception and labeling sidecars than as direct short-horizon forecasters

If we fine-tune anything next, the better order is:

1. fine-tune or train a smaller regime/event classifier
2. use GPT-OSS to generate structured labels for macro regime, event severity, crowd narrative, and branch explanations
3. only consider LoRA on GPT-OSS after the numeric stack has cleaner targets and aligned modalities

What LoRA on GPT-OSS is actually good for here:

- regime classification
- event severity labels
- crowd-mood labeling
- branch explanation and debate quality
- structured abstain / caution tags

What it is still not the best tool for:

- direct next-5-minute numeric prediction
- replacing the TFT / numeric simulator
- fixing low edge caused by weak price labels

## Best Next Execution Order

1. Rebuild and verify news/crowd embeddings so they are genuinely informative and timestamp-aligned.
2. Cloud retrain the numeric model against the redesigned multi-horizon, hold-aware targets.
3. Fold the regime analog output into retraining through sample weights, auxiliary targets, or a meta-label precision gate.
4. Run uncapped walk-forward evaluation and stricter filtered backtests after retraining.
5. Add a recent-regime-only training run so sparse old news/crowd history does not dilute the modern multimodal signal.
6. Train a separate regime / precision-gate model so the simulator can abstain more often instead of forcing weak predictions.
7. Only then revisit whether LLM fine-tuning is worth the cost.

Updated practical priority after the latest remote pass:

1. redesign the precision gate features and threshold search so abstention is neither all-or-nothing nor fully permissive
2. treat `15m / 30m` as first-class evaluation horizons instead of focusing almost entirely on `5m`
3. promote the next numeric head toward explicit horizon-specific confidence / hold outputs
4. add regime-conditioned training or routing instead of one shared head for every environment

Updated again after the hold/confidence pass:

1. run the next cloud pass on the new `12-output` head
2. redesign gate labels around real tradeability and minority-risk avoidance
3. evaluate strategic `15m / 30m` performance before spending more energy on raw `5m`
4. route UI / live decision summaries through the strategic horizon view instead of the old first-channel assumption

Updated again after the stricter abstention + richer macro/positioning expansion:

1. redesign the precision gate around true tradeability instead of letting hold-heavy windows dominate
2. make abstention much stricter and keep participation intentionally low
3. keep `15m / 30m` as the main simulator product story
4. expand official macro + positioning coverage before spending more GPU time on LLM fine-tuning

## Latest Cloud Pass On The 12-Output Head

The upgraded cloud run is now complete for both:

- `mh12_full`
- `mh12_recent`

This was the first real remote pass using:

- multi-horizon direction targets
- per-horizon hold targets
- per-horizon confidence targets
- strategic evaluation centered on `15m / 30m`
- the current precision gate

### Full-History Run (`mh12_full`)

Split:

- train: `2009-2020`
- validation: `2021-2023`
- test: `2024-2026`

Key test metrics:

- strategic ROC-AUC: `0.5133`
- `15m` ROC-AUC: `0.5125`
- `30m` ROC-AUC: `0.5139`
- `15m` hold ROC-AUC: `0.7142`
- `30m` hold ROC-AUC: `0.7124`
- strategic hold rate: `0.7650`
- strategic confidence mean: `0.2975`

Walk-forward / filtered backtest view:

- calibrated strategic ROC-AUC: `0.5098`
- participation rate: `0.6387`
- win rate: `0.6453`
- trade count: `504205`
- hold count: `285262`

Important interpretation:

- the direction edge is still weak
- the hold head is learning something materially better than the direction head
- the backtest win rate looks much better than ROC-AUC because the system is still structurally long-biased and class-imbalanced
- so this is not evidence of a true `64.5%` directional edge

### Recent-Regime Run (`mh12_recent`)

Split:

- train: `2021-2024`
- validation: `2025`
- test: `2026`

Key test metrics:

- strategic ROC-AUC: `0.5136`
- `15m` ROC-AUC: `0.5110`
- `30m` ROC-AUC: `0.5157`
- `15m` hold ROC-AUC: `0.6473`
- `30m` hold ROC-AUC: `0.6377`
- strategic hold rate: `0.5799`
- strategic confidence mean: `0.4334`

Walk-forward / filtered backtest view:

- calibrated strategic ROC-AUC: `0.4948`
- participation rate: `0.8559`
- win rate: `0.6485`
- trade count: `66062`
- hold count: `11123`

Important interpretation:

- the recent-regime model is slightly better on `30m` direction than the full-history run
- the recent-regime hold head is weaker than the full-history hold head
- the gate is still not meaningful enough, because it allows too much participation
- calibrated ROC-AUC falling below `0.5` in walk-forward is a warning that confidence calibration and thresholding still need work

### What These Runs Actually Tell Us

Good news:

- `15m / 30m` really are better targets than raw `5m`
- the hold/confidence redesign was the right architectural move
- the model is learning abstention-related structure better than pure direction

Bad news:

- the directional edge is still only slightly above chance
- the current precision gate is not selective enough
- backtest win rate is still too easy to overread because it is helped by bias / participation structure

So the next best move is still:

1. redesign gate labels around real tradeability and minority-risk avoidance
2. make abstention much more selective
3. treat `15m / 30m` as the primary product story
4. improve regime conditioning and analog scoring before chasing larger models

## Current In-Flight V2 Cloud Pass

There is now a newer cloud run in progress built around the lessons from `mh12_full` and `mh12_recent`.

The current remote tags are:

- `mh12_full_v2`
- `mh12_recent_v2`

What changed in this pass:

- the precision gate was redesigned around true strategic tradeability
- new gate features were added for:
  - strategic spread
  - strategic tradeability
- threshold search now targets lower participation instead of letting the gate stay almost always-on
- fusion was rebuilt with `15m` as the primary horizon
- hold-row weight was lowered to reduce over-holding bias
- macro inputs were expanded with more official series
- positioning inputs were expanded with much broader historical CFTC coverage

Current remote status:

- fast data refresh completed
- perception rebuild completed
- fusion rebuild completed
- the `v2` cloud pass completed and artifacts are now local

Latest facts from that refreshed data pass:

- news aligned coverage ratio: about `0.0097`
- crowd aligned coverage ratio: about `0.5633`
- primary fusion horizon: `15m`
- rebuilt primary hold rate: about `0.4908`
- rebuilt sample-weight mean: about `1.4698`

Important interpretation:

- official macro + positioning context is now materially better represented than before
- crowd/positioning coverage is much stronger than news coverage
- this makes the next `15m / 30m` pass more meaningful than the earlier `5m`-anchored runs

### Final V2 Result

`mh12_full_v2`:

- strategic ROC-AUC: `0.5142`
- `15m` ROC-AUC: `0.5153`
- `30m` ROC-AUC: `0.5141`
- `15m` hold ROC-AUC: `0.6749`
- `30m` hold ROC-AUC: `0.6784`
- walk-forward calibrated ROC-AUC: `0.5117`
- backtest trade count: `0`
- backtest participation: `0.0`

`mh12_recent_v2`:

- strategic ROC-AUC: `0.5086`
- `15m` ROC-AUC: `0.5046`
- `30m` ROC-AUC: `0.5082`
- `15m` hold ROC-AUC: `0.6226`
- `30m` hold ROC-AUC: `0.6157`
- walk-forward calibrated ROC-AUC: `0.4954`
- backtest trade count: `0`
- backtest participation: `0.0`

What this means:

- the richer macro / positioning context helped the `15m / 30m` framing slightly
- the new gate logic overshot and became too strict
- the next problem is no longer “make abstention stricter”
- the next problem is “restore low but non-zero participation without losing selectivity”

So the next concrete step is:

1. keep the `15m / 30m` framing
2. keep the richer macro / positioning inputs
3. relax and recalibrate the gate
4. only then run capital-based `$10 / $1000` backtests once real trades exist again

### Local Relaxed-Gate Re-Evaluation On Synced `v2` Artifacts

After the strict `v2` runs landed with zero trades, the local evaluation stack was upgraded so synced cloud manifests can be reused properly:

- remote checkpoint paths now resolve to local `models/tft/*`
- remote precision-gate paths now resolve to local `models/tft/*`
- local walk-forward calibration now falls back to the available local years when the manifest's original validation years are missing
- gate-threshold search now uses score quantiles from the actual learned gate distribution instead of hardcoded threshold bands
- capital-style backtests now include both:
  - compounding `R`-multiple view
  - fixed-risk `R`-multiple view

Important local-data caveat:

- the current local fused tensor / timestamp bundle only covers `2026`
- so these relaxed reruns are useful for gate recovery and account-style inspection
- but they are not replacements for the original full cloud walk-forward runs

Local relaxed rerun result on `mh12_full_v2`:

- calibration source years: `2026`
- optimized thresholds:
  - decision threshold: `0.53`
  - confidence floor: `0.06`
  - hold threshold: `0.55`
  - gate threshold: `0.1240`
- participation rate: `0.1500`
- win rate: `0.5614`
- trades: `1124`
- fixed-risk backtest:
  - `$10`: final capital about `37.60`, return about `276.0%`, max drawdown about `42.9%`
  - `$1000`: final capital about `3760.0`, return about `276.0%`, max drawdown about `42.9%`

Local relaxed rerun result on `mh12_recent_v2`:

- calibration source years: `2026`
- optimized thresholds:
  - decision threshold: `0.53`
  - confidence floor: `0.06`
  - hold threshold: `0.55`
  - gate threshold: `0.1824`
- participation rate: `0.1500`
- win rate: `0.6085`
- trades: `1124`
- fixed-risk backtest:
  - `$10`: final capital about `58.80`, return about `488.0%`, max drawdown about `27.9%`
  - `$1000`: final capital about `5880.0`, return about `488.0%`, max drawdown about `27.9%`

How to interpret these:

- the gate is no longer dead
- low but non-zero participation is back
- the recent-regime `v2` model still looks stronger than the full-history `v2` model on the local `2026` slice
- the capital curves are still `R`-multiple simulations, not broker-grade execution backtests
- the next real test should happen on the cloud again after retraining, not only on the reduced local artifact slice

## Concrete Ways To Break The Low-50% Ceiling

These are the most realistic paths to beat the current roughly `51%` regime.

1. Stop optimizing for every bar.
   Train for selective precision on meaningful moves and let the system hold / abstain much more often.

2. Split the problem by regime.
   A single model for calm chop, macro shock, and strong trend is too blunt. Add a regime classifier and route scoring by regime.

3. Train recent-regime and full-history models separately.
   The recent years have much better news/crowd coverage. A modern multimodal model and a long-history price-only model should probably not be forced into one identical training pool.

4. Use a meta-label precision gate.
   Let one model predict direction and another decide whether the setup is worth trusting at all.

5. Promote the supervision head to multi-horizon.
   The artifacts are now ready for `5m / 10m / 15m / 30m`; the next architectural gain is making the model output those horizons directly.

6. Make backtesting harsher, not softer.
   Add slippage, spread assumptions, and stronger confidence floors so weak apparent edge disappears early instead of fooling us.

7. Use GPT-OSS for labels, not for raw price forecasting.
   LoRA can still help if it improves regime labels, event severity, discussion polarity, and abstain logic. That can materially help the numeric stack without pretending the LLM is a market oracle.

8. Improve the gate before improving model size.
   Right now the abstention path is a bigger bottleneck than model scale. Better trade selection is more likely to help than a larger predictor.

## On Retraining Again And Again

Blindly retraining the same model again and again is usually not how the edge improves.

Why:

- if labels are weak, more retraining mostly teaches the same noise again
- if the split is weak, more retraining often means overfitting
- if the gate is weak, more retraining only makes the same bad decisions look more confident
- if the signal is stronger at `15m / 30m` than `5m`, retraining a `5m`-obsessed setup just keeps wasting compute

What repeated retraining is good for:

- after changing labels
- after changing horizons
- after improving modalities
- after improving regime routing
- after improving abstention logic

So the correct loop is:

1. improve the problem definition
2. retrain
3. walk-forward test
4. inspect failure modes
5. change the system again
6. retrain

That is very different from simply rerunning the same training command and hoping accuracy drifts upward.

## No-Auth Data Sources Worth Adding Next

These are the most promising public sources to expand the project without adding account friction.

Highest-priority additions:

- GDELT event / mention / GKG feeds for broader geopolitical and finance context:
  `https://www.gdeltproject.org/data.html`
- FRED API for rates, dollar, inflation, spreads, and macro state:
  `https://fred.stlouisfed.org/docs/api/fred/`
- U.S. Treasury daily yield curve archives for cleaner rate-curve features:
  `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives`
- CFTC historical Commitment of Traders archives for positioning / crowd proxy:
  `https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm`
- BLS developer/API resources for CPI and labor/event context:
  `https://www.bls.gov/developers/`
- Federal Reserve FOMC historical calendars and statements:
  `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
- ECB Data Portal API for FX and euro-area macro context:
  `https://data.ecb.europa.eu/help/api/overview`
- World Gold Council Goldhub data and research for structural gold context:
  `https://www.gold.org/goldhub/data`

Why these matter more than random extra datasets:

- they add macro regime context
- they add positioning context
- they add event / policy context
- they are more likely to improve `15m / 30m` strategic simulation quality than generic sentiment noise

Lower-priority / already partially covered:

- Google News RSS query feeds
- Yahoo chart endpoints
- Stooq-style market context mirrors

These are still useful, but the bigger gain now is richer macro + positioning + event-state structure, not just more headline volume.

What was already wired in after the latest expansion:

- extra FRED series:
  - `DGS2`
  - `T10Y2Y`
  - `BAMLH0A0HYM2`
  - `NFCI`
- official Treasury archive CSVs:
  - par yield curve archive
  - par real yield curve archive
  - real long-term rate archive
- Federal Reserve raw policy pages:
  - `fomccalendars.htm`
  - yearly FOMC press-release index pages for `2023-2026`
- extra GDELT slices:
  - central-bank gold reserve headlines
  - rates / yield-shock headlines
  - energy / inflation headlines
- persona macro logic now reads:
  - front-end rates
  - curve shape
  - credit stress
  - financial conditions
  - payroll / unemployment context
- historical CFTC financial futures archives were expanded across many years instead of only the most recent files
- historical CFTC disaggregated archives were expanded across `2010-2026`

Current local raw-data footprint after the latest pull:

- total files under `data/raw`: `89`
- total size: about `48.0 MB`
- macro: `19` files / about `3.0 MB`
- news: `14` files / about `1.0 MB`
- crowd / positioning: `56` files / about `43.9 MB`

## Current V3 Run

## Latest Local V4 Preparation

The next local upgrade after `v3` is now in place and is aimed at making abstention more meaningful instead of simply making the gate tighter.

What was implemented in this pass:

- quant context was expanded with:
  - `quant_regime_persistence`
  - `quant_state_entropy`
  - `quant_tail_risk`
- fusion now writes a dedicated gate-context matrix:
  - `data/features/gate_context.npy`
- gate context currently includes:
  - ATR / BB-width / volume state
  - session overlap
  - regime strength
  - regime persistence
  - transition risk
  - state entropy
  - tail risk
  - volatility realism
  - fair-value dislocation
  - trend score
  - state imbalance
  - chop risk
- target construction now uses the new quant context directly when deciding:
  - hold rows
  - sample weights
  - strategic tradeability
- precision-gate training now accepts aligned context features instead of learning only from horizon agreement
- walk-forward evaluation now accepts aligned gate-context rows and applies them to:
  - calibration-time gate scoring
  - per-fold gate scoring
  - full backtest scoring
- old saved precision gates remain backward-compatible:
  - if a gate was trained before context features existed, the evaluator now falls back gracefully to the base gate feature set instead of erroring
- compounding capital backtests were hardened:
  - log-equity math is now used internally
  - absurd long-run overflow now reports as `overflowed: true`
  - fixed-risk capital mode remains the safest human-readable mode

Latest local rebuild after that pass:

- quant rebuild completed on the local synced slice:
  - rows: `7611`
  - state count: `4`
  - feature columns: `14`
- fused rebuild completed and now writes:
  - `sample_weights.npy`
  - `target_hold_mask.npy`
  - `targets_multihorizon.npz`
  - `gate_context.npy`

Validation from this pass:

- targeted compile checks passed
- full local test suite passed:
  - `51/51`
- smoke multi-horizon training with the new gate-context path completed successfully:
  - run tag: `gate_ctx_smoke`
  - output dim: `12`
  - gate artifact written successfully
  - gate context sample artifact written successfully
- local walk-forward smoke against the existing `mh12_full_v3` checkpoint also ran successfully with the backward-compatible gate logic

Important local smoke result using the updated evaluator and the existing `mh12_full_v3` checkpoint:

- year: `2026`
- capped windows: `250`
- decision threshold: `0.53`
- confidence floor: `0.06`
- optimized gate threshold: about `0.1878`
- trades: `11`
- participation: `4.4%`
- win rate: `45.45%`

Interpretation:

- the new gate-context path is functioning
- older saved checkpoints and gates can now be evaluated under the new walk-forward stack
- the local synced slice is still too small and too recent to treat this as a meaningful edge estimate
- the main value of this pass is infrastructure correctness for the next cloud run, not the tiny local PnL

## Quant Ideas Worth Trying Next

These are the strongest quantitative-model additions still worth trying around the simulator.

Highest-priority next additions:

1. Hidden-regime router

- use HMM / Markov-switching / semi-Markov logic
- route:
  - branch expansion
  - branch pruning
  - gate aggressiveness
  - persona weights

2. Volatility realism model

- use GARCH / EGARCH or simpler realized-volatility forecasting
- penalize branches that project impossible `15m / 30m` moves
- feed the realism score into:
  - branch fitness
  - final cone width
  - abstention

3. Kalman / state-space fair-value model

- estimate a latent gold fair value from:
  - DXY
  - yields
  - VIX
  - crude / inflation proxies
- use dislocation from fair value as:
  - institutional / whale bias input
  - branch realism penalty
  - reversal probability helper

4. Meta-label gradient-boosted gate

- add LightGBM / XGBoost on top of:
  - simulator outputs
  - TFT outputs
  - analog scores
  - quant context
  - bot disagreement
- predict:
  - tradeable / not tradeable
  - expected precision bucket

5. Quantile / conformal cone calibration

- make the cone match empirical error bands instead of only branch dispersion
- this is one of the best ways to make the simulator honest even before raw ROC-AUC improves

6. Change-point detection

- use offline or online change-point logic to detect:
  - post-news regime breaks
  - transition from trend to chop
  - transition from calm to panic
- feed that into abstention and branch pruning

7. Positioning factor model

- build cleaner CFTC-based crowding / squeeze features
- especially useful for `15m / 30m` when macro and positioning line up

8. Dependency / spread models

- dynamic spreads:
  - gold vs DXY
  - gold vs real yields
  - gold vs VIX
  - gold vs silver ratio
- use these as institutional sanity checks rather than standalone signals

Recommended order:

1. quant-aware gate on the cloud
2. meta-label boosted gate
3. volatility realism model
4. hidden-regime router
5. fair-value Kalman layer
6. conformal cone calibration

Why this order:

- the gate and realism layers are most likely to improve filtered win quality fastest
- they also preserve the identity of Nexus Trader as a simulator rather than turning it into a generic black-box predictor

The cloud `v3` cycle is now in its late stage with the quant-hybrid layer and the refreshed macro / news / positioning inputs.

Primary goals of `v3`:

- keep `15m / 30m` as the main story
- use richer macro / positioning context
- keep GPT as a sidecar only
- restore low-but-nonzero participation after the over-strict `v2` gate
- test whether the quant-hybrid layer improves strategic ROC-AUC and filtered backtests

Current `v3` cloud pipeline steps:

1. fast dataset refresh
2. macro rebuild
3. news embeddings rebuild
4. crowd embeddings rebuild
5. quant context build
6. persona outputs rebuild
7. fused artifacts rebuild
8. tests
9. `mh12_full_v3` training
10. `mh12_full_v3` walk-forward evaluation
11. `mh12_recent_v3` training
12. `mh12_recent_v3` walk-forward evaluation

Latest confirmed `v3` runtime facts:

- remote full quant rebuild completed successfully
- remote persona rebuild completed successfully
- remote fusion rebuild completed successfully
- remote tests passed
- `mh12_full_v3` training completed successfully
- `mh12_full_v3` walk-forward and backtest completed successfully
- `mh12_recent_v3` training completed successfully
- the active remote stage is now `walkforward_mh12_recent_v3`
- current remote fused rows: `6,024,602`
- current primary horizon: `15m`
- current fused hold rate: about `0.5078`
- current fused sample-weight mean: about `1.9238`
- current quant transition risk: about `0.1271`
- current quant volatility realism: about `0.8096`
- GPU memory usage during training has been around `91%`
- latest observed GPU package power during training has ranged roughly `267W - 299W`

Latest confirmed `mh12_full_v3` training metrics:

- strategic test ROC-AUC: about `0.5090`
- strategic test accuracy: about `0.5140`
- `15m` test ROC-AUC: about `0.5096`
- `30m` test ROC-AUC: about `0.5088`
- `15m` hold ROC-AUC: about `0.6797`
- `30m` hold ROC-AUC: about `0.6830`

Latest confirmed `mh12_full_v3` walk-forward / backtest metrics:

- years: `2024, 2025, 2026`
- calibration source years: `2021, 2022, 2023`
- optimized gate threshold: about `0.1643`
- overall calibrated strategic ROC-AUC: about `0.5077`
- `15m` calibrated ROC-AUC remains roughly in the low `0.51` range by fold
- `30m` calibrated ROC-AUC remains roughly in the low `0.51` range by fold
- trades: `146,756`
- participation: about `18.59%`
- win rate: about `63.95%`
- fixed-risk `$10` final capital: about `8196.40`
- fixed-risk `$1000` final capital: about `819640.0`
- fixed-risk max drawdown: about `30.34%`

Important backtest caveat:

- the compounding capital mode is numerically exploding to `NaN` / absurd magnitudes on long runs
- the fixed-risk `R`-multiple mode is the only remotely interpretable capital view right now
- these are still simulation-style backtests, not broker-grade execution backtests with spread/slippage

Latest confirmed `mh12_recent_v3` training metrics:

- strategic test ROC-AUC: about `0.4985`
- `15m` test ROC-AUC: about `0.5017`
- `30m` test ROC-AUC: about `0.5019`
- recent-regime directional training did not improve beyond the full-history `v3` pass
- the recent-regime walk-forward / backtest stage is still running, so final trade counts for that branch are not available yet

Interpretation so far:

- the quant-hybrid pass is still helping the hold / abstention side more than the directional side
- `15m / 30m` remain the right horizons to optimize around
- the full-history `v3` backtest is now materially active rather than dead-zero, which is progress
- the recent-regime `v3` training did not beat the full-history `v3` training on raw directional metrics
- the next remaining unknown is whether `mh12_recent_v3` walk-forward/backtest produces better filtered participation or capital behavior than `mh12_full_v3`

Important operational note:

- an overlapping duplicate `v3` launch was detected and cleaned up
- the active remote run is now the detached, logged pipeline written to:
  - `outputs/logs/remote_v3_pipeline.log`
- a local watcher is now polling the cloud run every `60s` and will auto-sync:
  - `outputs/evaluation`
  - `models/tft`
  - `outputs/logs`
- a local post-sync summarizer is also waiting to generate:
  - `outputs/evaluation/v3_summary.json`
  - `outputs/evaluation/v3_summary.md`

## Current V4 Prep State

A new local `v4` preparation pass has now been implemented and validated before the next cloud cycle.

What changed locally in this pass:

- the evaluator was recovered and upgraded so it understands:
  - `gate_context.npy`
  - the saved precision gate
  - the new boosted meta gate
  - combined gate scores during walk-forward and backtesting
- the gate is now quant-aware at a deeper level, not only agreement-aware:
  - routed regime confidence now flows into the gate context
  - Kalman fair-value dislocation now flows into the gate context
  - state entropy / tail-risk / persistence remain part of the abstention stack
- the branch layer was upgraded so routed-regime and Kalman signals affect synthetic forward rows and branch realism scoring
- the cone layer now uses weighted quantile bands rather than only mean-plus-width expansion
- the compounding capital backtest now reports overflow cleanly instead of collapsing into meaningless `NaN`
- a boosted meta gate was added through `src/training/meta_gate.py`
  - preferred backends: XGBoost, LightGBM
  - fallback backend: sklearn HistGradientBoosting
- the trainer now saves both:
  - linear precision gate
  - boosted meta gate
- older saved gates remain backward-compatible in the evaluator

Files materially updated in this pass:

- `src/quant/hybrid.py`
- `src/pipeline/fusion.py`
- `src/mcts/tree.py`
- `src/mcts/reverse_collapse.py`
- `src/mcts/cone.py`
- `src/evaluation/walkforward.py`
- `src/training/meta_gate.py`
- `scripts/train_fused_tft.py`
- `tests/test_quant_hybrid.py`
- `tests/test_mcts.py`
- `tests/test_walkforward.py`

Local validation status after the recovery/upgrade:

- full unit test suite: `51/51` passing again
- repaired evaluator compiles successfully
- repaired trainer compiles successfully
- quant/router/Kalman/cone modules compile successfully

Important interpretation:

- this pass was mainly about making the next cloud run smarter and safer
- it does not yet prove a better remote edge by itself
- the next meaningful proof step is a true `v4` remote run on the Jupyter ROCm box

Jupyter connectivity note:

- during this chat, direct Jupyter API reachability from the local restricted path was inconsistent
- after escalation, the server endpoint responded successfully again
- so the cloud `v4` pass should be launched from this repaired local code state, not from the older `v3` state

Planned `v4` cloud objective:

1. sync the repaired local `v4` code to the cloud repo/workspace
2. rebuild quant context and fused artifacts on the full remote dataset
3. run remote tests
4. train `mh12_full_v4`
5. run remote walk-forward + backtest for `2024, 2025, 2026`
6. optionally run `mh12_recent_v4`
7. compare:
   - strategic ROC-AUC
   - `15m` ROC-AUC
   - `30m` ROC-AUC
   - hold quality
   - participation
   - trade count
   - fixed-risk capital results

Current remote `v4` execution status in this chat:

- validated local `v4` files were synced to the Jupyter workspace root `nexus/`
- an initial detached remote `v4` process was launched successfully but failed at `train_mh12_full_v4` because `scripts/train_fused_tft.py` was missing the `META_GATE_PATH` import
- that trainer import mismatch was fixed locally, revalidated, and re-uploaded
- the active remote process is now the resumed `v4` training pipeline
- current remote pid file points to: `outputs/logs/remote_v4_resume.pid`
- current remote main log is: `outputs/logs/remote_v4_resume.log`
- a local monitor process is polling the remote run tags:
  - `mh12_full_v4`
  - `mh12_recent_v4`
- the local monitor will auto-sync finished artifacts back into:
  - `outputs/evaluation/`
  - `models/tft/`
  - `outputs/logs/`
- a local waiting summarizer is also running for:
  - `outputs/evaluation/v4_summary.json`
  - `outputs/evaluation/v4_summary.md`

GPT-OSS role remains unchanged:

- GPT-OSS should still be added after these numeric/quant builds as a sidecar
- valid GPT-OSS jobs here are:
  - macro/news/crowd interpretation
  - structured labels
  - branch explanations
  - swarm judgment
- GPT-OSS should not replace the numeric forecaster or the quant gate

## Open-Source Comparative Review

I reviewed the projects under [SimilarExistingSolutions](C:/PersonalDrive/Programming/AiStudio/nexus-trader/SimilarExistingSolutions) to identify which production patterns Nexus should borrow from mature open-source systems.

Projects reviewed:

- `nautilus_trader-develop`
- `freqtrade-develop`
- `zipline-master`
- `backtrader-master`
- `pysystemtrade-develop`
- `TradeMaster-1.0.0`
- `TradingAgents-main`
- `AI-Trader-main`
- `awesome-systematic-trading-main`

Highest-value conclusions:

- `NautilusTrader` is the best model for deterministic event-driven backtest architecture and research/live parity.
- `Freqtrade` is the best model for validation tooling, especially lookahead-bias detection, recursive-analysis, and threshold-search ergonomics.
- `Zipline` and `Backtrader` are the best models for slippage, commission, fill realism, and trade lifecycle accounting.
- `pysystemtrade` is the best model for production operations discipline, backups, diagnostics, and scheduled system workflows.
- `TradeMaster` is the strongest reference for market-dynamics labeling and regime-supervision ideas.
- `TradingAgents` and `AI-Trader` are useful for the GPT sidecar, debate memory, and market-intel aggregation, but not for the numeric prediction core.
- `awesome-systematic-trading` is a reference list, not an implementation source.

What Nexus should implement from this review:

- a proper `src/backtest/` module with event-driven semantics
- slippage / fee / fill abstractions
- a Nexus-specific `lookahead-analysis` command
- a Nexus-specific recursive feature / leakage analysis command
- a structured backtest result object instead of ad hoc report assembly
- explicit market-dynamics label generation for regime supervision
- daily diagnostics / backup / rebuild scripts
- better GPT-sidecar debate memory and market-intel snapshot caching

What Nexus should not copy:

- LLM as the primary numeric forecaster
- generic copy-trading marketplace patterns
- RL-first core prediction architecture

The full implementation breakdown from this review is now documented in:

- [SIMILAR_EXISTING_SOLUTIONS_IMPLEMENTATION_PLAN.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/SIMILAR_EXISTING_SOLUTIONS_IMPLEMENTATION_PLAN.md)

## Codebase Areas To Audit Next

High-priority files to audit in future chats:

- `src/service/live_data.py`
- `src/service/app.py`
- `src/simulation/abm.py`
- `src/simulation/personas.py`
- `src/mcts/tree.py`
- `src/mcts/reverse_collapse.py`
- `src/mcts/cone.py`
- `src/training/train_tft.py`
- `src/ui/web.py`

Questions to ask during that audit:

- are branch prices realistic?
- is branch scoring meaningful?
- is confidence calibrated or inflated?
- do personas differ enough to matter?
- are we leaking future information?
- is the UI comparing like-for-like timestamps and sources?
- are technical overlays helping or just adding noise?

## Master Prompt For Future Chats

Use this to continue the project in a new conversation:

```text
We are working in the repo:
C:\PersonalDrive\Programming\AiStudio\nexus-trader

Read these files first:
1. MASTER.md
2. CONTEXT_HANDOFF.md
3. PROJECT_MASTER_SUMMARY.md
4. TODO_NEXT.md

Project identity:
Nexus Trader is a market simulator, not a simple trading bot.
It is supposed to model multiple trader personas, branch many plausible futures, reverse-collapse those branches into a consensus, and display a live probability cone plus predicted-vs-actual comparison UI.

Important constraints:
- Do not pretend the system already has true high accuracy.
- Preserve interpretability.
- Treat disagreement as signal.
- Keep numeric prediction separate from LLM-based text reasoning.
- Use local LLMs only as sidecars for macro/news/crowd interpretation, branch explanation, and structured labeling.

Current important truths:
- Local UI/API are already running through FastAPI.
- Essential cloud artifacts already exist locally, including the TFT checkpoint and evaluation summaries.
- XAUUSD now uses a spot-calibrated public gold feed instead of raw futures-only display.
- The old unrealistic 5-minute projection bug was fixed by switching to weighted branch-path price aggregation.
- Branches now roll forward their own synthetic state instead of replaying the same base row.
- GPT-OSS is integrated as a local sidecar.
- GPT receives a 120-bar chart snapshot summary in the live context path.
- the LLM route can now be switched between LM Studio local and Ollama cloud.
- The UI includes manual technical-analysis panels and branch overlays.
- The UI includes a 10-bot specialist swarm, a GPT judge panel, a swarm graph, and a branch graph view.
- GPT judgment now reads raw news, public discussion, technical structure, branches, and bot outputs together.
- The UI includes a TradingView desk panel for external chart confirmation.
- The compare chart is moving toward a multi-horizon final forecast path rather than a 5-minute-only raw simulator line.
- The branch layer now uses short-window regime analog context instead of only single-row analog similarity.
- The project still needs deeper MiroFish-like branching, branch fitness scoring, reverse-confidence collapse, and better confidence calibration.

Highest-value next work:
1. Rebuild and verify news/crowd embeddings plus timestamp alignment in the fused artifacts.
2. Expand the quant layer beyond clustering into regime routing, volatility realism penalties, and fair-value-aware branch scoring.
3. Retrain on the cloud with short-window analog regime features, quant-hybrid context, and multi-horizon hold-aware targets.
4. Improve reverse collapse and minority scenario logic.
5. Build a meta-label precision gate so the system can abstain when edge is weak without collapsing to zero trades.
6. Connect specialist bots more directly into branch scoring and multi-horizon consistency checks.
7. Expand walk-forward tests and stricter filtered backtests beyond capped smoke runs.
8. Add cone hit-rate and branch-tracking evaluation into the reporting loop.

When working, inspect the existing codebase first and explain what is already implemented before changing architecture.
```

## Operational Notes

- The cloud Jupyter contents API is slow for large data syncs.
- Fast sync strategy:
  - keep local copies of `models/` and `outputs/`
  - use GitHub for code persistence
  - do not rely on file-by-file Jupyter pulls for huge raw datasets

## Final Honest Summary

Nexus Trader is now a partially implemented live market simulator with:

- local hosting
- spot-calibrated XAU feed
- branching and reverse-collapse primitives
- routed `v5` predictor heads with regime diagnostics
- timestamped predicted-vs-actual UI
- GPT-OSS sidecar integration
- manual technical-analysis panels
- 10 specialist strategy bots
- GPT swarm judge
- GPT final manual stance
- swarm and branch graph views
- TradingView desk panel
- preserved local model artifacts

But it is still not the full MiroFish-style simulator the user originally wanted.

Current honest performance summary:

- raw ROC-AUC is still only around `0.50x`
- filtered `15m/30m` simulator backtests are holding in roughly the mid-`60%` win-rate regime
- the most credible edge still comes from abstention, branch selection, and regime-aware filtering

The next serious step is not “bigger random model.”
The next serious step for `v6` is:

- finish the cloud `v6` run and compare it honestly against `v5`
- train a real historical branch-selector dataset where branches are labeled by similarity to actual future paths
- learn branch-selector weights instead of leaning mainly on fallback scoring
- add branch-level volatility rejection directly into ranking and collapse
- use historical retrieval priors more strongly in branch survival probability
- add richer order-flow / liquidity proxies if reliable data becomes available
- only after that deepen GPT-OSS as the structured catalyst / explanation sidecar

## Latest Open-Source-Informed Roadmap Note

The previous "bigger model" framing should be treated as outdated.

The better next step is:

- keep the predictor and selector architecture branch-first
- improve realism and validation before scaling model size
- implement production-grade backtest semantics, slippage, fees, and fills
- add lookahead-analysis and recursive leakage checks
- add explicit market-dynamics labeling for regime supervision
- keep GPT-OSS as a sidecar for reasoning, labeling, and explanation, not as the numeric predictor

See:

- [SIMILAR_EXISTING_SOLUTIONS_IMPLEMENTATION_PLAN.md](C:/PersonalDrive/Programming/AiStudio/nexus-trader/SIMILAR_EXISTING_SOLUTIONS_IMPLEMENTATION_PLAN.md)

## Latest Open-Source-Informed Implementation Pass

The first real implementation wave from the open-source review is now in the repo.

Implemented in this pass:

- new reusable backtest package:
  - `src/backtest/engine.py`
  - `src/backtest/fees.py`
  - `src/backtest/slippage.py`
  - `src/backtest/results.py`
  - `src/backtest/validation.py`
- `src/evaluation/walkforward.py` now delegates the directional backtest and capital backtests through the extracted backtest engine instead of keeping all of that logic inline
- first leakage / artifact-audit commands:
  - `scripts/lookahead_analysis.py`
  - `scripts/recursive_feature_analysis.py`
- first dedicated tests for the new layer:
  - `tests/test_backtest_engine.py`
  - `tests/test_backtest_validation.py`

What this gives Nexus now:

- explicit fee and slippage abstractions instead of only raw unit-PnL assumptions
- structured trade-record capable backtests
- a place to grow toward event-driven fills and richer trade lifecycle semantics
- a first real lookahead / leakage audit path over fused artifacts
- a first recursive window-consistency audit path

Current local validation for this pass:

- `python -m unittest tests.test_backtest_engine tests.test_backtest_validation tests.test_walkforward -v`
- all passing

Current audit-script status:

- `lookahead_analysis.py`
  - runs successfully on current local fused artifacts
  - current local report does not show suspicious raw feature-target correlations
- `recursive_feature_analysis.py`
  - currently skips cleanly on this machine because `data/features/fused_tensor.npy` is not present locally
  - the command is ready once fused tensor artifacts are rebuilt locally or on the cloud

Immediate next implementation step from this new base:

- upgrade `src/backtest/` from cost-aware scoring to true event-driven fill simulation
- add execution realism modules inspired by Zipline / Backtrader
- deepen the audit scripts into full Freqtrade-style lookahead and recursive analysis over model artifacts and selector features

## Latest Event-Driven Backtest And Artifact-Audit Pass

The next open-source-informed wave is now also implemented.

Added in this pass:

- event-driven backtest primitives:
  - `src/backtest/events.py`
  - `src/backtest/event_engine.py`
- stronger artifact audit layer:
  - `src/backtest/artifact_audit.py`
  - `scripts/model_artifact_leakage_analysis.py`
- new tests:
  - `tests/test_event_backtest.py`
  - `tests/test_artifact_audit.py`

What this adds beyond the first pass:

- explicit market bars, simulated orders, and fill events
- a true event-driven directional backtest path instead of only direct unit-PnL scoring
- artifact-level auditing for:
  - training summary vs walk-forward consistency
  - gate participation problems
  - gate context / timestamp alignment
  - manifest / checkpoint traceability

Current verified status:

- `python -m unittest tests.test_event_backtest tests.test_artifact_audit tests.test_backtest_engine tests.test_backtest_validation tests.test_walkforward -v`
- all passing

Current local artifact-audit finding:

- the default local `training_summary.json` / `walkforward_report.json` path still shows `gate_participation_near_zero`
- that means the generic local default artifacts are still weaker / older than the stronger tagged cloud runs like `v5` and `v6`
- this is useful because the new audit now catches that inconsistency directly

Best next move from this new state:

- wire the event-driven backtest mode into tagged walk-forward reports
- add branch-selector and gate-artifact audits for specific tagged runs like `mh12_full_v6`
- begin implementing market-dynamics labeling from the TradeMaster-inspired roadmap

## Latest Event-Driven Walkforward Integration

The next open-source-informed step is now implemented locally.

What changed:

- `src/evaluation/walkforward.py` now emits both:
  - classic directional backtests
  - event-driven execution-aware backtests
- `scripts/run_walkforward_evaluation.py` now saves:
  - `backtest`
  - `event_driven_backtest`
  - `event_driven_by_horizon`
- event-driven evaluation uses real aligned OHLC rows from local `price_features.csv`
- event-driven evaluation now applies non-zero costs:
  - `FixedBpsFeeModel`
  - `VolatilityScaledSlippageModel`
- older local environments now degrade cleanly when:
  - sklearn-backed meta gates are unavailable
  - old precision-gate artifacts are feature-incompatible
  - CPU evaluation sees `bfloat16` tensors

Important honesty:

- the event-driven execution-aware view is much harsher than the old unit-PnL summary
- this is useful
- it means the previous framework was still flattering the system more than real fill semantics do

Local smoke result after the event-driven integration:

- `run_tag`: `mh12_full_v6`
- years: `2026`
- capped windows: `256`
- classic directional backtest:
  - win rate: about `53.91%`
  - avg unit pnl: about `0.0781`
- event-driven strategic backtest:
  - win rate: about `26.56%`
  - avg unit pnl: about `-0.4704`
- event-driven by horizon:
  - `5m` win rate: about `20.86%`
  - `10m` win rate: about `22.27%`
  - `15m` win rate: about `23.23%`
  - `30m` win rate: about `26.56%`

Interpretation:

- the selector/gate/model stack still looks much better in direction-only scoring than in cost-aware execution-aware replay
- `30m` is still the least bad horizon in the event-driven smoke, which reinforces the longer-horizon product story
- this is exactly why the event-driven backtest layer needed to be wired in before `v7`

## Latest Tagged Artifact Audit

The new tagged artifact audit has now also been applied to `mh12_full_v6`.

Output:

- `outputs/evaluation/model_artifact_leakage_report_mh12_full_v6.json`

Key findings:

- `precision_gate_has_high_train_accuracy_but_low_train_precision`
- `gate_participation_near_zero`

Interpretation:

- the saved gate artifact still looks overfit / weakly tradeable on the local synced slice
- the stronger filtered backtest regime from cloud runs should not be treated as sufficient proof by itself
- the next training cycle needs better regime supervision and better selector training, not just more epochs

## Latest Market-Dynamics Label Builder

The TradeMaster-inspired market-dynamics labeling path is now implemented.

Added:

- `src/regime/__init__.py`
- `src/regime/labeling.py`
- `scripts/build_market_dynamics_labels.py`
- `tests/test_market_dynamics_labeling.py`

Current local label artifact:

- `data/processed/market_dynamics_labels.csv`
- `outputs/evaluation/market_dynamics_report.json`

Current local dominant label distribution on the synced slice:

- `breakout`: about `62.28%`
- `low_volatility`: about `17.50%`
- `range`: about `10.33%`
- `high_volatility`: about `7.16%`

Important note:

- local parquet engines were unavailable in this environment
- so the label builder falls back cleanly to CSV instead of failing

## Current Best Next Step Toward V7

The right next step is now clearer:

1. fold `market_dynamics_labels` into the next training feature / supervision path
2. use the event-driven backtest as a first-class report, not only the classic directional view
3. retrain only after those labels are integrated into:
   - regime routing
   - selector scoring
   - gate supervision
4. then launch a true `v7` cloud run and judge it primarily on:
   - event-driven `15m / 30m` behavior
   - tagged artifact-audit health
   - participation vs drawdown vs win rate

## Latest V7 Prep Integration

The next local step toward `v7` is now implemented and validated.

What changed:

- market-dynamics labels are now merged into fused-artifact generation
- market-dynamics signals now directly influence:
  - primary hold mask expansion
  - sample-weight construction
  - gate-context construction
- the fused-artifact builder now records the dynamics source path in `fusion_report.json`
- training summaries / manifests now also record the market-dynamics artifact path
- local meta-gate training now degrades cleanly when boosted backends are unavailable instead of failing the whole run

Files updated in this pass:

- `src/pipeline/fusion.py`
- `scripts/build_fused_artifacts.py`
- `scripts/train_fused_tft.py`
- `src/training/meta_gate.py`
- `tests/test_fusion_pipeline.py`
- `tests/test_walkforward.py`

Latest local fused-artifact summary after the dynamics merge:

- rows: `7611`
- feature width: `100`
- target hold rate: about `26.84%`
- sample-weight mean: about `2.9408`
- average dynamics confidence: about `0.5217`
- average breakout probability: about `0.4265`
- average range-risk proxy: about `0.1696`
- average panic-risk proxy: about `0.1181`

Latest local smoke training result:

- run tag: `v7_dynamics_smoke`
- output head: `12`
- smoke checkpoint saved:
  - `models/tft/final_tft_v7_dynamics_smoke.ckpt`
- smoke manifest saved:
  - `models/tft/model_manifest_v7_dynamics_smoke.json`
- smoke precision gate saved:
  - `models/tft/precision_gate_v7_dynamics_smoke.json`

Important interpretation:

- the new dynamics-aware path trains end to end locally
- so `v7` is now technically ready to launch on the cloud
- but the real success condition for `v7` should still be:
  - better event-driven `15m / 30m` behavior
  - healthier tagged artifact audits
  - less divergence between classic directional backtests and event-driven backtests

## Current Best Next Step

The clean next move from this repo state is:

1. rebuild market-dynamics labels on the cloud full dataset
2. rebuild fused artifacts there with the new dynamics-aware weighting and gate context
3. launch `mh12_full_v7`
4. run walk-forward with the event-driven backtest path active
5. compare `v7` against `v6` on:
   - raw `15m / 30m` ROC-AUC
   - classic directional backtest
   - event-driven backtest
   - tagged artifact-audit findings

## Latest V7 Cloud Execution State

The real `v7` cloud cycle is now launched from this repo state.

What was uploaded to the Jupyter workspace for `v7`:

- `scripts/remote_v7_train.py`
- `scripts/launch_remote_v7_jupyter.py`
- `scripts/check_remote_v7_status_jupyter.py`
- `scripts/build_market_dynamics_labels.py`
- `scripts/build_fused_artifacts.py`
- `scripts/train_fused_tft.py`
- `scripts/run_walkforward_evaluation.py`
- `scripts/model_artifact_leakage_analysis.py`
- `src/regime/*`
- `src/backtest/*`
- `src/pipeline/fusion.py`
- `src/training/meta_gate.py`
- `src/evaluation/walkforward.py`
- updated tests required by the remote `unittest discover` stage

Remote execution details:

- Jupyter workspace root: `nexus/`
- remote launcher script: `scripts/launch_remote_v7_jupyter.py`
- remote pipeline script: `scripts/remote_v7_train.py`
- remote pid file: `outputs/logs/remote_v7_pipeline.pid`
- remote log: `outputs/logs/remote_v7_pipeline.log`
- latest confirmed remote pid at launch: `823500`
- latest confirmed remote log tail right after launch:
  - `===== build_quant_context =====`

The `v7` pipeline shape is:

1. `build_quant_context`
2. `build_persona_outputs`
3. `build_market_dynamics_labels`
4. `build_fused_artifacts`
5. remote tests
6. `train_mh12_full_v7`
7. `walkforward_mh12_full_v7`
8. `audit_mh12_full_v7`
9. `train_mh12_recent_v7`
10. `walkforward_mh12_recent_v7`
11. `audit_mh12_recent_v7`

This is the first cloud run where the intended comparison target is explicitly:

- classic filtered backtest
- event-driven execution-aware backtest
- artifact-audit health

and not only the older flattering directional summary.

Local long-run monitoring is also active now:

- watcher log:
  - `outputs/logs/monitor_cloud_run_v7.out.log`
- watcher errors:
  - `outputs/logs/monitor_cloud_run_v7.err.log`
- waiting summarizer log:
  - `outputs/logs/summarize_v7_results_wait.out.log`
- waiting summarizer errors:
  - `outputs/logs/summarize_v7_results_wait.err.log`

Local helper added for final `v7` reporting:

- `scripts/summarize_v7_results.py`

That summarizer is designed to generate:

- `outputs/evaluation/v7_summary.json`
- `outputs/evaluation/v7_summary.md`

once both:

- `mh12_full_v7`
- `mh12_recent_v7`

have synced down locally together with their:

- training summaries
- walk-forward reports
- backtest reports
- artifact-audit reports

## Latest V7 Failure And Resume State

The first `v7` cloud attempt did not finish cleanly.

What failed:

- `build_quant_context`: completed
- `build_persona_outputs`: completed
- `build_market_dynamics_labels`: completed
- `build_fused_artifacts`: completed
- remote tests: completed
- `train_mh12_full_v7`: completed far enough to move on
- `walkforward_mh12_full_v7`: failed

Failure cause:

- `src/evaluation/walkforward.py` only resolved event-driven price artifacts from:
  - `data/features/price_features.parquet`
  - `data/features/price_features.csv`
- the cloud run was actually using the processed legacy artifact under:
  - `data_store/processed/XAUUSD_1m_features.parquet`
- so event-driven evaluation raised:
  - `FileNotFoundError: Price feature artifact is required for event-driven evaluation.`

What was fixed:

- `src/evaluation/walkforward.py`
  - event-driven price resolution now also falls back to:
    - `LEGACY_PRICE_FEATURES_PARQUET`
    - `LEGACY_PRICE_FEATURES_CSV`

To avoid wasting GPU / compute time, the pipeline was resumed from the failed stage only instead of restarting from the top.

New resume helpers added:

- `scripts/remote_v7_resume.py`
- `scripts/launch_remote_v7_resume_jupyter.py`
- `scripts/check_remote_v7_resume_status_jupyter.py`

Resume state at the latest confirmed check:

- remote resume pid: `832630`
- remote resume log:
  - `outputs/logs/remote_v7_resume.log`
- latest confirmed resume tail:
  - `===== walkforward_mh12_full_v7 =====`
- resume process status:
  - `running`

Resume pipeline shape:

1. `walkforward_mh12_full_v7`
2. `audit_mh12_full_v7`
3. `train_mh12_recent_v7`
4. `walkforward_mh12_recent_v7`
5. `audit_mh12_recent_v7`

Important note:

- the original watcher was still waiting because it only watches for completed JSON artifacts
- direct Jupyter status checks are the source of truth for remote failure / resume state

## Current V7 Success Criteria

`v7` should be judged by these, in order:

1. event-driven `15m / 30m` backtest quality
2. trade count / participation vs drawdown
3. tagged artifact-audit findings
4. raw `15m / 30m` ROC-AUC

Important honesty:

- if `v7` only improves the classic directional summary while event-driven results stay poor, that is not a true win
- if `v7` preserves or improves event-driven `15m / 30m` behavior with cleaner audit findings, that is a real step forward even if raw ROC-AUC is still only modestly improved

## Final V7 Results

`v7` is now completed and synced locally.

Final summary artifacts:

- `outputs/evaluation/v7_summary.json`
- `outputs/evaluation/v7_summary.md`
- `outputs/evaluation/training_summary_mh12_full_v7.json`
- `outputs/evaluation/walkforward_report_mh12_full_v7.json`
- `outputs/evaluation/backtest_report_mh12_full_v7.json`
- `outputs/evaluation/model_artifact_leakage_report_mh12_full_v7.json`
- `outputs/evaluation/training_summary_mh12_recent_v7.json`
- `outputs/evaluation/walkforward_report_mh12_recent_v7.json`
- `outputs/evaluation/backtest_report_mh12_recent_v7.json`
- `outputs/evaluation/model_artifact_leakage_report_mh12_recent_v7.json`

`mh12_full_v7`

- strategic ROC-AUC: `0.5079`
- strategic accuracy: `0.6439`
- `15m` ROC-AUC: `0.5142`
- `30m` ROC-AUC: `0.5135`
- classic filtered backtest:
  - trades: `110,037`
  - participation: `13.94%`
  - win rate: `56.83%`
  - avg unit pnl: `0.0190`
- event-driven backtest:
  - trades: `110,037`
  - participation: `14.53%`
  - win rate: `45.37%`
  - avg unit pnl: `-0.0128`
- event-driven by horizon:
  - `5m`: win rate `34.03%`, avg unit pnl `-0.0504`
  - `10m`: win rate `44.97%`, avg unit pnl `-0.0141`
  - `15m`: win rate `51.68%`, avg unit pnl `0.0047`
  - `30m`: win rate `46.00%`, avg unit pnl `-0.0114`
- audit findings:
  - `precision_gate_has_high_train_accuracy_but_low_train_precision`

`mh12_recent_v7`

- strategic ROC-AUC: `0.4988`
- strategic accuracy: `0.6436`
- `15m` ROC-AUC: `0.5051`
- `30m` ROC-AUC: `0.5063`
- classic filtered backtest:
  - trades: `8,462`
  - participation: `10.96%`
  - win rate: `57.94%`
  - avg unit pnl: `0.0174`
- event-driven backtest:
  - trades: `8,462`
  - participation: `10.96%`
  - win rate: `47.65%`
  - avg unit pnl: `-0.0053`
- event-driven by horizon:
  - `5m`: win rate `32.64%`, avg unit pnl `-0.0388`
  - `10m`: win rate `45.82%`, avg unit pnl `-0.0095`
  - `15m`: win rate `53.48%`, avg unit pnl `0.0076`
  - `30m`: win rate `47.57%`, avg unit pnl `-0.0054`
- audit findings:
  - none

Important final `v7` interpretation:

- the raw predictor is still weak by ROC-AUC and remains near the `0.50x` regime
- the classic filtered backtest still looks materially better than the execution-aware one
- the event-driven path is the real bar now, and on that bar:
  - `15m` is the only clearly positive horizon in both full and recent `v7`
  - `30m` did not hold up as well as hoped once execution realism was applied
- this is still a simulator-first system, but `v7` strongly suggests the best current product story is:
  - selective `15m` future simulation
  - strict abstention elsewhere

Best post-`v7` direction:

1. optimize explicitly for event-driven `15m` quality rather than broad strategic accuracy
2. improve branch-selector realism and tradeability supervision
3. continue using GPT-OSS as sidecar / judge, not numeric forecaster
4. treat classic filtered backtests as secondary to execution-aware reports

## Current V8 Architecture Direction

`v8` is the first version where the project is being pushed toward a formal `generator -> selector` stack instead of only refining the generator and then filtering its outputs afterward.

The intended `v8` architecture is:

```text
Current Market State
  -> Formal Regime Features
  -> Volatility Envelope / Plausibility Features
  -> Fair-Value / Dislocation Features
  -> Historical Analog Retrieval
  -> TFT Generates Candidate Futures
  -> Branch Archive Builder
  -> Learned Branch Selector
  -> Final Future Path + Rejection Rationale
```

The major `v8` additions in local code are:

- `src/v8/hmm_regime.py`
- `src/v8/garch_volatility.py`
- `src/v8/fair_value.py`
- `src/v8/analog_retrieval.py`
- `src/v8/branch_selector_v8.py`

Supporting `v8` scripts now exist for:

- quant-stack building
- branch-archive construction
- branch-selector training
- selector evaluation
- final `v8` summary generation

Important design intent:

- TFT is treated as the imagination engine
- the new selector stack is being trained to decide which candidate future is most realistic
- `15m` remains the primary horizon of interest
- event-driven evaluation remains the real benchmark

## Current V8 Cloud Run Status

`v8` is not finished yet, but the cloud run is healthy and actively training.

Current remote run:

- remote pipeline script: `scripts/remote_v8_train.py`
- remote launcher script: `scripts/launch_remote_v8_jupyter.py`
- remote status helper: `scripts/check_remote_v8_status_jupyter.py`
- remote runtime helper: `scripts/check_remote_v8_runtime_jupyter.py`
- latest confirmed remote pid: `949242`
- latest confirmed active stage: `train_mh12_full_v8`

What has already completed remotely in this `v8` cycle:

1. `build_quant_context`
2. `build_persona_outputs`
3. `build_market_dynamics_labels`
4. `build_fused_artifacts`
5. `build_v8_quant_stack`
6. remote tests

Current runtime truth from the cloud box:

- parent process is alive
- child training workers are active
- GPU utilization is now high instead of the old underfed state
- latest confirmed ROCm snapshot during `mh12_full_v8` training:
  - GPU use: `96%`
  - power draw: about `296W`
  - VRAM allocated: about `91%`

Important systems interpretation:

- the old complaint about `15% - 30%` GPU usage does not describe the current `v8` run
- the current `v8` training is saturating the MI300X much better
- the persistent `91% VRAM` reading is still mostly allocator reservation behavior and is not by itself a problem
- the quiet remote log does not mean the run is idle; the runtime helper confirms active compute

Current local `v8` artifacts already present:

- `outputs/evaluation/v8_quant_stack_report.json`
- `outputs/evaluation/v8_branch_archive_report_v8_smoke.json`
- `outputs/evaluation/v8_branch_selector_report_v8_smoke.json`
- `outputs/evaluation/v8_evaluation_report_v8_smoke.json`

Current local interpretation:

- the `v8` stack works locally in smoke mode
- the full cloud result still needs to finish before any honest `v8` performance summary can be made

## Expected V8 Finish Line

The cloud `v8` run should eventually produce:

- `outputs/evaluation/training_summary_mh12_full_v8.json`
- `outputs/evaluation/walkforward_report_mh12_full_v8.json`
- `outputs/evaluation/v8_branch_archive_report_mh12_full_v8.json`
- `outputs/evaluation/v8_branch_selector_report_mh12_full_v8.json`
- `outputs/evaluation/v8_evaluation_report_mh12_full_v8.json`
- `outputs/evaluation/training_summary_mh12_recent_v8.json`
- `outputs/evaluation/walkforward_report_mh12_recent_v8.json`
- `outputs/evaluation/v8_branch_archive_report_mh12_recent_v8.json`
- `outputs/evaluation/v8_branch_selector_report_mh12_recent_v8.json`
- `outputs/evaluation/v8_evaluation_report_mh12_recent_v8.json`
- `outputs/evaluation/v8_summary.json`
- `outputs/evaluation/v8_summary.md`

The real `v8` success criteria are:

1. selector beats generator baseline on branch choice
2. event-driven `15m` improves versus `v7`
3. top-3 containment is materially better than top-1 random baseline
4. analog / regime / fair-value features show real importance in the selector
5. the final chosen path is more realistic than the older probability-only collapse

## Final V8 Results

`v8` is now completed and synced locally.

Important note about completion:

- the cloud pipeline itself completed all major research steps
- the remote final summarizer failed only because the cloud workspace did not have `outputs/evaluation/v7_summary.json`
- the real `v8` artifacts were still produced successfully
- local summary generation succeeded after sync

Final local summary artifacts:

- `outputs/evaluation/v8_summary.json`
- `outputs/evaluation/v8_summary.md`
- `outputs/evaluation/v8_evaluation_report_mh12_full_v8.json`
- `outputs/evaluation/v8_evaluation_report_mh12_recent_v8.json`
- `outputs/evaluation/v8_branch_selector_report_mh12_full_v8.json`
- `outputs/evaluation/v8_branch_selector_report_mh12_recent_v8.json`
- `outputs/evaluation/v8_branch_archive_report_mh12_full_v8.json`
- `outputs/evaluation/v8_branch_archive_report_mh12_recent_v8.json`

### `mh12_full_v8`

Core walk-forward / backtest state:

- walk-forward strategic ROC-AUC: `0.5103`
- classic filtered backtest win rate: `56.92%`
- event-driven backtest win rate: `46.12%`
- event-driven `15m`:
  - win rate: `52.39%`
  - avg unit pnl: `0.008563`

Branch-selector results:

- selector top-1 branch accuracy: `0.61325`
- selector top-3 containment: `0.63650`
- average selected path error: `0.000769`
- generator baseline path error: `0.000774`
- selector error improvement: `0.000006`
- selector event-driven `15m`:
  - trade count: `4000`
  - win rate: `52.20%`
  - avg unit pnl: `0.045599`
  - cumulative unit pnl: `182.396896`

Interpretation:

- `full_v8` is the strongest `v8` branch
- branch selection is learning something real at the top-1 / top-3 level
- the selector only slightly improves path error versus the generator baseline
- the strongest useful product story remains selective `15m`

### `mh12_recent_v8`

Core walk-forward / backtest state:

- walk-forward strategic ROC-AUC: `0.4972`
- classic filtered backtest win rate: `58.31%`
- event-driven backtest win rate: `47.77%`
- event-driven `15m`:
  - win rate: `53.57%`
  - avg unit pnl: `0.006001`

Branch-selector results:

- selector top-1 branch accuracy: `0.5800`
- selector top-3 containment: `0.6032`
- average selected path error: `0.001451`
- generator baseline path error: `0.001451`
- selector error improvement: effectively `0.0`
- selector event-driven `15m`:
  - trade count: `2500`
  - win rate: `50.72%`
  - avg unit pnl: `0.014306`
  - cumulative unit pnl: `35.764118`

Interpretation:

- `recent_v8` did not beat the stronger recent `v7` event-driven `15m` win-rate profile
- branch selection works structurally, but the selector is still too close to the generator baseline
- the recent-regime branch selector needs denser / more selective archive construction to become meaningfully different

## Honest V8 Interpretation

What `v8` did accomplish:

- it turned branch selection into a real learned problem
- it produced a functioning branch archive and selector-training pipeline
- it achieved materially non-random top-1 branch accuracy on held-out selector evaluation
- it preserved the main project truth that `15m` is the only horizon with consistent positive execution-aware behavior

What `v8` did not accomplish yet:

- it did not create a dramatic raw ROC-AUC breakout
- it did not make HMM / analog / fair-value features obviously dominant in the selector
- it did not create a large selector-vs-generator separation on path error
- it did not improve cone containment enough to call the future-path cone trustworthy yet

Best honest summary:

- `v8` is a major architectural success
- `v8` is only a modest predictive success
- the biggest win is that Nexus now has a true learned branch-selector research path instead of only generator-first heuristics

## V9 Direction

The best next move after `v8` is:

1. promote the learned selector into the live branch ranking path
2. rebuild recent branch archives with denser sampling around high-impact regimes
3. optimize directly for event-driven `15m` selected-branch pnl / drawdown / containment
4. use GPT-OSS only for explanation and catalyst rationale around the chosen branch, not as the numeric selector

## Latest Gate Redesign

After the local Backtrader week checks, one more failure mode became clear:

- pure ungated `v8` overtrades and loses
- the saved full `v8` gate is too strict on the local synced slice and often takes `0` trades
- the old gate logic was effectively applying a double abstention penalty:
  - hard binary tradeability labels during gate training
  - then a threshold optimizer that still preferred very cautious participation

The gate has now been redesigned locally to be softer and more quant-aware instead of just stricter.

Files changed:

- `src/training/train_tft.py`
- `src/training/meta_gate.py`
- `src/evaluation/walkforward.py`
- `tests/test_walkforward.py`

What changed in the new gate:

- tradeability labels are now soft instead of hard all-or-nothing masks
- quant context is converted into a continuous `quant_tradeability_score` instead of strict rule cutoffs
- gate inference now blends:
  - learned logistic gate score
  - quant tradeability score
- the precision/meta gate combiner is softer and no longer punishes moderate scores as harshly
- walk-forward threshold search now considers much lower gate quantiles and targets non-dead participation more explicitly

Important local verification:

- `tests.test_walkforward` passes
- `scripts/validate_pipeline.py` passes

Local recalibration sanity check on the synced `mh12_full_v8` slice:

- recalibrated gate threshold: about `0.5631`
- recalibrated gate participation: about `16.01%`
- recalibrated gate train precision: about `50.09%`
- recalibrated applied positive rate matches train participation exactly after the inference/training blend bug was fixed

Important interpretation:

- this redesign does **not** mean `v8` suddenly became profitable
- it does mean the gate is now being pushed toward a usable middle zone instead of the old binary failure:
  - `everything passes`
  - or `nothing passes`

Practical next step from here:

1. regenerate the saved precision/meta gate artifacts under this softer quant-aware logic
2. rerun local `15m` execution-aware checks first
3. then use the redesigned gate in the next `v9` / selector-focused cycle

Local compatibility note:

- rerunning `mh12_full_v8` with the redesigned gate *application* but the old saved `v8` gate artifact no longer collapsed to `0` trades
- but it became effectively too permissive on the local week slice, which means:
  - the new gate scoring scale and the old saved gate threshold are no longer a trustworthy match
- conclusion:
- the code redesign is correct
- the persisted gate artifacts now need to be regenerated under the new logic before any honest full-stack comparison is made

## Current Gate Refresh Status

The `v8` gate-refresh pass has now been launched on the cloud Jupyter server instead of doing a full retrain.

Purpose:

- keep the existing `mh12_full_v8` and `mh12_recent_v8` checkpoints
- regenerate:
  - `precision_gate_*`
  - `meta_gate_*`
- rerun walk-forward / backtest so the saved gate thresholds match the new softer gate scale

New scripts added for this refresh:

- `scripts/regenerate_gate_artifacts.py`
- `scripts/remote_v8_gate_refresh.py`
- `scripts/launch_remote_v8_gate_refresh_jupyter.py`
- `scripts/check_remote_v8_gate_refresh_status_jupyter.py`

Current remote status at handoff:

- remote PID: `1231842`
- remote log: `/home/rocm-user/jupyter/nexus/outputs/logs/remote_v8_gate_refresh.log`
- current confirmed stage: `regenerate_gate_mh12_full_v8`

Important note:

- this is **not** a full `v8` retrain
- it is a much cheaper recalibration pass over the saved `v8` checkpoints and artifacts

## Local V8 Direct Manual Terminal Update

Date:

- `2026-04-04T11:52:43+05:30`

What was completed locally:

- redesigned the local dashboard UI into a dark glass / soft-neumorphic terminal
- exposed `Signal Stack` selection directly in the UI
- defaulted the local operator flow to `V8 Direct Manual`
- kept `LM Studio Local` as the primary local sidecar route
- added top status-header pills for:
  - local session
  - market status
  - model mode
  - latency
  - live/manual mode
  - LLM route
- added dedicated manual-trading hero cards and restored the missing `LLM Persona Tilts` panel

Files changed:

- `src/ui/web.py`
- `src/service/app.py` (already had the `v8_direct` path and was validated)

Local runtime status:

- local server started successfully in `v8_direct` mode
- URL: `http://127.0.0.1:8000/ui`
- server PID: `22872`
- local logs:
  - `outputs/logs/local_v8_direct_server.out.log`
  - `outputs/logs/local_v8_direct_server.err.log`

Smoke verification:

- `/ui` serves the redesigned terminal shell
- `/api/simulate-live?symbol=XAUUSD&llm_provider=lm_studio&stack_mode=v8_direct` returns:
  - `stack_mode = v8_direct`
  - `manual_trading_mode = true`
  - `final_forecast.mode = v8_direct`
  - `ensemble_prediction.mode = v8_direct`

Operational meaning:

- this local terminal is now running `V8 only` for manual desk usage
- external gate veto is not being used in this local operating mode
- the user is expected to take trades manually from the projected path / final judge / structure view rather than allow auto-filtered execution

## Current Local V9 Bootstrap Status

Date:

- `2026-04-04`

What has now been implemented locally for `v9`:

- `src/v9/branch_labels.py`
- `src/v9/branch_features_v9.py`
- `src/v9/selector_torch.py`
- `src/v9/persona_calibration.py`
- `src/v9/memory_bank.py`
- `src/v9/contradiction_detector.py`
- `src/v9/regret_gate.py`
- `src/v9/selector_experiments.py`
- `scripts/build_v9_branch_dataset.py`
- `scripts/train_v9_selector_torch.py`
- `scripts/train_v9_memory_bank.py`
- `scripts/run_v9_selector_experiments.py`

What has been wired into the live / evaluation stack already:

- local Python environment now uses `torch 2.11.0+cu128`
- local runtime now sees:
  - `NVIDIA GeForce RTX 4070`
- `src/service/live_data.py` now exposes:
  - persisted `persona_weights`
  - memory-bank context if encoder/index artifacts exist
  - contradiction classification
- `src/evaluation/walkforward.py` and `src/training/meta_gate.py` now accept a `v9` regret-aware gate score in the combined gate path

Current local `v9` artifacts produced:

- `outputs/v9/branch_labels.parquet`
- `outputs/v9/branch_features_v9.parquet`
- `outputs/v9/branch_features_v9.report.json`
- `outputs/v9/branch_labels_recent_v8.parquet`
- `outputs/v9/branch_features_recent_v8.parquet`
- `outputs/v9/branch_features_recent_v8.report.json`
- `outputs/v9/selector_torch_recent.json`
- `outputs/v9/selector_experiment_results_recent.json`
- `outputs/v9/selector_experiment_results_recent.md`
- `outputs/v9/memory_bank_report.json`
- `checkpoints/v9/memory_bank/encoder.pt`
- `checkpoints/v9/memory_bank/index.npz`

Important local verification:

- `tests.test_v9_branch_dataset`: passes
- `tests.test_v9_runtime`: passes
- `tests.test_v8_stack`: passes
- `tests.test_walkforward`: passes

Important local `v9` bootstrap findings:

- the `v8` branch archive converts cleanly into the new `v9` label / feature artifacts
- local branch consensus is still extremely concentrated:
  - `outputs/v9/branch_features_v9.report.json`
  - mean consensus strength is about `0.99997`
- cone containment from the current archive-derived `v9` labels is still effectively `0.0`
- minority rescue remains nearly absent on the current archive
- this strongly supports the earlier diagnosis that branch diversity is still too weak before any honest cone-quality claim can be made

Current local memory-bank bootstrap:

- local CUDA smoke training completed successfully
- report:
  - device: `cuda`
  - window size: `60`
  - sample count: `504`
  - embedding dim: `64`
  - epochs: `2`
  - validation accuracy: about `0.5248`

Important honesty about the current local selector-experiment results:

- the newly generated `outputs/v9/selector_experiment_results_recent.json` numbers are **local bootstrap experiment numbers**
- they are useful for architecture plumbing and feature separability checks
- they are **not** yet the final honest `v9` benchmark
- they are mostly in-sample on the local recent archive feature frame, so they must not be compared directly to the final walk-forward `v8` headline numbers as if they were equivalent

What the local selector bootstrap did show:

- simple tree-based selectors can rank the current recent archive very aggressively on the local frame
- the torch selector path now trains and runs on CUDA successfully
- despite strong local ranking numbers on some experiment variants, event-style selected-branch quality remains close to flat on this bootstrap frame
- this again suggests the main bottleneck is still branch realism / diversity and not just selector capacity

Current best `v9` interpretation:

- the `v9` architecture pass is now genuinely underway locally
- the runtime scaffolding for:
  - persona calibration
  - memory bank
  - contradiction detection
  - regret-aware gating
  - selector experiments
  is now present in the repo
- but `v9` is **not** complete yet
- the current state is best described as:
  - `full local V9 architecture bootstrap in progress`

Immediate next `v9` priorities from this state:

1. train a denser local memory bank on a larger fused slice and feed its confidence back into selector experiments
2. improve selector evaluation so top-3 / event-driven / cone metrics are measured on stricter held-out folds
3. connect contradiction and persona-calibration outputs more explicitly into the live branch ranking story
4. decide whether the branch archive itself must be regenerated with stronger diversity before further selector optimization is worth more time

## Latest Local V9 Follow-Through

What was completed after the first `v9` bootstrap:

- trained a denser local CUDA memory bank again
- added `scripts/enrich_v9_features_with_memory_bank.py`
- generated memory-bank-enriched branch feature artifacts for:
  - recent `v8` archive
  - full `v8` archive
- rewrote `v9` selector experiments to use a stricter held-out split by `sample_id`
  - train: first `80%`
  - validation: last `20%`
- upgraded the live branch-ranking path so `v9` runtime context now influences ranking more directly through:
  - memory-bank alignment
  - contradiction-aware confidence scaling
  - persona-weight alignment

New `v9` artifacts now present:

- `outputs/v9/branch_features_recent_v8_enriched.parquet`
- `outputs/v9/branch_features_v9_enriched.parquet`
- `outputs/v9/selector_experiment_results_recent_heldout.json`
- `outputs/v9/selector_experiment_results_recent_heldout.md`
- `outputs/v9/selector_experiment_results_full_heldout.json`
- `outputs/v9/selector_experiment_results_full_heldout.md`
- `outputs/v9/selector_experiment_results_full_heldout_enriched.json`
- `outputs/v9/selector_experiment_results_full_heldout_enriched.md`

Important local held-out `v9` result:

- these numbers are now materially more honest than the earlier local in-frame bootstrap because they are evaluated on a held-out chronological `sample_id` slice
- they are still local archive experiments, not the final end-to-end walk-forward `v9` benchmark

Recent held-out `v9` selector read:

- validation samples: `1,200`
- strongest current recent held-out selector:
  - `selector_d`
  - top-1 branch accuracy: `0.7200`
  - top-3 containment: `0.7200`
  - event-driven `15m` win rate: `0.5000`
  - event-driven `15m` avg unit pnl: about `-0.00003`

Full held-out `v9` selector read:

- validation samples: `1,600`
- strongest current full held-out selector:
  - `selector_d`
  - top-1 branch accuracy: `0.756875`
  - top-3 containment: `0.756875`
  - event-driven `15m` win rate: `0.50875`
  - event-driven `15m` avg unit pnl: about `0.000037`

Important interpretation of these held-out results:

- held-out branch ranking quality is now genuinely stronger than the old weak local bootstrap numbers
- recent held-out `top-1/top-3` now sits near the lower edge of the intended `v9` target zone
- full held-out `top-1/top-3` is stronger still and reaches the mid-`0.75x` range
- but the execution-style `15m` result remains effectively flat
- this is the most important current truth:
  - selector ranking is improving
  - tradeable branch realism is **not** improving in proportion

What memory-bank enrichment showed:

- adding memory-bank-enriched branch features did **not** materially improve the strongest tree-based held-out selectors yet
- the analog-heavy selector remains weaker than the full feature selectors
- the torch selector path is now reproducible and materially healthier than before, but it still does not beat the stronger tree-based held-out selectors
- conclusion:
  - memory bank is now architecturally integrated
  - but it is not yet producing a clear measurable lift on held-out branch selection

Current best honest `v9` interpretation after the held-out pass:

- `v9` is making real progress on selector architecture and evaluation honesty
- `v9` is **not yet** solving the deeper product problem
- the dominant bottleneck still appears to be:
  - branch diversity
  - cone realism
  - tradeability supervision
- the current archive still behaves like a system where:
  - branches can often be ranked
  - but the ranked branches are still too similar and too weakly tradeable

Best next `v9` move from here:

1. regenerate or diversify the branch archive rather than spending too much more time squeezing the current selector
2. preserve the stronger held-out selector evaluation harness that now exists
3. keep memory bank and contradiction logic in the live path, but treat them as support signals until they show measurable held-out lift
4. make cone containment and minority-branch usefulness first-class targets in the next archive / generator upgrade

## V10 Local Generator-First Pass

What was completed locally for the first full `v10` pass:

- implemented a new `src/v10` package for:
  - diversity audit
  - regime-conditioned temperature scheduling
  - diversity-regularized branch scoring
  - minority branch guarantee
  - cone supervision
  - archive diversification / regeneration
- added local scripts for:
  - `scripts/audit_branch_diversity.py`
  - `scripts/regenerate_branch_archive_v10.py`
  - `scripts/retrain_selector_from_v10_archive.py`
- extended `config/project_config.py` with `v10` artifact paths
- upgraded the local torch selector path so it can train on variable branch counts with padding + masks

Important `v10` local artifacts now present:

- `outputs/v10/branch_diversity_recent_baseline.json`
- `outputs/v10/branch_diversity_full_baseline.json`
- `outputs/v10/branch_archive_v10_recent.parquet`
- `outputs/v10/branch_archive_v10_recent.report.json`
- `outputs/v10/branch_archive_v10_full.parquet`
- `outputs/v10/branch_archive_v10_full.report.json`
- `outputs/v10/branch_labels_v10_full.parquet`
- `outputs/v10/branch_features_v10_full.parquet`
- `outputs/v10/selector_torch_v10_full.pt`
- `outputs/v10/selector_experiment_results_v10_full.json`
- `outputs/v10/selector_experiment_results_v10_full.md`

### V10 Phase 0

Prerequisite artifacts were verified locally before the generator-first pass:

- `outputs/v9/branch_features_v9_enriched.parquet`
- `outputs/v9/branch_features_recent_v8_enriched.parquet`
- `outputs/v9/selector_experiment_results_full_heldout.json`
- `checkpoints/v9/memory_bank/encoder.pt`
- `checkpoints/v9/memory_bank/index.npz`
- `models/tft/final_tft.ckpt`
- `data/features/fused_features.npy`

### V10 Phase 1 Baseline Diversity Audit

Recent baseline audit from `outputs/v8/branch_archive_mh12_recent_v8.parquet`:

- mean consensus strength: about `0.99999999`
- mean minority share: `0.0`
- mean direction std: `0.0`
- cone containment rate: about `0.0238`
- full-path containment rate: `0.0`

Full baseline audit from `outputs/v8/branch_archive_mh12_full_v8.parquet`:

- mean consensus strength: about `0.99997076`
- mean minority share: about `0.000029`
- mean direction std: about `0.000108`
- cone containment rate: about `0.02675`
- full-path containment rate: `0.0`

This is the clearest local confirmation yet that the pre-`v10` generator was still collapsing into nearly one-sided cones.

### V10 Phases 2-6 Regenerated Archive Result

Recent regenerated `v10` archive result:

- source archive:
  - `6,000` samples
  - `384,000` branch rows
- regenerated archive:
  - `57,919` branch rows
- regenerated mean consensus strength: about `0.6879`
- regenerated mean minority share: about `0.3121`
- regenerated mean direction std: about `0.8941`
- regenerated cone containment rate: about `0.3993`
- regenerated full-path containment rate: about `0.2630`

Full regenerated `v10` archive result:

- source archive:
  - `8,000` samples
  - `512,000` branch rows
- regenerated archive:
  - `77,817` branch rows
- regenerated mean consensus strength: about `0.6928`
- regenerated mean minority share: about `0.3072`
- regenerated mean direction std: about `0.8904`
- regenerated cone containment rate: about `0.3979`
- regenerated full-path containment rate: about `0.2610`

Best honest reading of these regeneration numbers:

- `v10` materially fixed the fake-thin cone problem at the archive level
- the generator is now producing branch sets with real directional spread instead of near-total consensus
- cone containment improved by roughly an order of magnitude versus the pre-`v10` archive baseline
- the system now carries a genuine minority branch in aggregate rather than almost none

### V10 Phase 7+ Selector Follow-Through

The local selector was retrained on the regenerated full `v10` archive using the existing `v9` label / feature stack, now with variable branch-count support in the torch trainer.

Important local `v10` selector read from `outputs/v10/selector_experiment_results_v10_full.json`:

- held-out validation samples: `1,600`
- strongest current `top-3` selector:
  - `selector_e`
  - top-1 branch accuracy: `0.441875`
  - top-3 branch containment: `0.86875`
  - event-driven `15m` win rate: `0.480625`
- torch selector on the new variable-width archive:
  - device: `cuda`
  - top-1 branch accuracy: `0.19375`
  - top-3 containment: `0.413125`
  - event-driven `15m` win rate: `0.486875`

Important interpretation:

- after `v10`, branch ranking still works and top-3 containment is now very strong on the regenerated archive
- the generator-side diversity problem is materially improved
- but event-style `15m` tradeability is still not convincingly positive yet
- so `v10` appears to be fixing the correct bottleneck first, but it has not yet converted that improvement into strong execution quality

Current best honest `v10` state:

- `phase 0` through `phase 7+` have now been executed locally in the first real `v10` archive cycle
- the generator-side diversity thesis was validated strongly
- the next step should not be to abandon `v10`
- the next step should be to keep the regenerated archive path and push the supervision toward:
  - tradeability
  - realized event outcomes
  - regime-specific execution quality

Important local verification for this pass:

- `tests.test_v10_generator`: passes
- `tests.test_v9_branch_dataset`: passes
- `tests.test_mcts`: passes

## V11 Execution-Translation Pass

What was implemented locally for the first `v11` research cycle:

- `SETL` in:
  - `src/v11/setl.py`
- `PCOP` in:
  - `src/v11/path_conditioned_outcome.py`
- `CESM` in:
  - `src/v11/crowd_state_machine.py`
- `PMWM` in:
  - `src/v11/persistent_world_model.py`
- integrated research backtest pipeline in:
  - `src/v11/research_backtest.py`
  - `scripts/run_v11_research_backtest.py`

Support changes added:

- `config/project_config.py` now contains `v11` artifact paths
- `src/v9/selector_torch.py` already supports variable branch counts from the `v10` archive, which remains important for `v11`
- `tests/test_v11_research.py` now covers the `v11` research layer

Important `v11` artifacts now present:

- `outputs/v11/research_backtest_full.json`
- `outputs/v11/research_backtest_full.md`
- `checkpoints/v11/selector_ranker.pkl`
- `checkpoints/v11/pcop_stage5.pkl`
- `checkpoints/v11/pcop_stage10.pkl`
- `checkpoints/v11/setl_regressor.pkl`

### What Each V11 Component Does

`SETL`:

- trains a second-stage expected-PnL model on selected branches
- predicts whether the selected branch is actually worth trading after costs
- converts branch realism into execution quality

`PCOP`:

- scores branch survival after the first `5m` or `10m` of actual path is known
- reweights branches by path consistency instead of leaving the original cone frozen
- supports delayed entry after confirmation

`CESM`:

- maps each sample into a crowd-emotional state
- current state space used locally:
  - `disbelief`
  - `greed`
  - `euphoria`
  - `panic`
  - `relief`
- the current local held-out fold only surfaced `disbelief` and `panic` strongly

`PMWM`:

- rolls a persistent world state across chronologically ordered samples
- carries forward:
  - institutional positioning estimate
  - retail sentiment momentum
  - structural memory strength
  - regime persistence
  - smart-money fingerprint
  - narrative age

### V11 Local Backtest Result

Input artifact used:

- `outputs/v10/branch_features_v10_full.parquet`

Split used:

- train samples: `6,400`
- validation samples: `1,600`
- chronological held-out validation fraction: `0.20`

Optimized `SETL` threshold learned on train:

- threshold: about `0.192485`
- train participation target found: about `0.35`

Held-out `v11` variant results:

`selector_only_open`:

- participation: `1.0000`
- win rate: `0.483125`
- avg unit pnl: `-0.030309`
- cumulative unit pnl: `-48.494495`

`setl_open`:

- participation: `0.360625`
- win rate: `0.641248`
- avg unit pnl: `0.286758`
- cumulative unit pnl: `165.459625`

`pcop_5m_setl`:

- participation: `0.256875`
- win rate: `0.535280`
- avg unit pnl: `0.072040`
- cumulative unit pnl: `29.608454`

`pcop_10m_setl`:

- participation: `0.264375`
- win rate: `0.529551`
- avg unit pnl: `0.060515`
- cumulative unit pnl: `25.597652`

`full_v11` staged policy:

- participation: `0.555000`
- win rate: `0.609234`
- avg unit pnl: `0.220903`
- cumulative unit pnl: `196.162125`
- stage usage on held-out set:
  - open: `712`
  - `pcop_5m`: `449`
  - `pcop_10m`: `439`

Best honest interpretation:

- this is the first local cycle where Nexus meaningfully separated:
  - plausible future
  - tradeable setup
- `SETL` appears to solve the immediate execution-translation gap much better than raw selector choice alone
- `PCOP` adds real value as a delayed-confirmation mechanism, though in this first local pass it is weaker than pure `SETL` at bar zero when used alone
- the strongest held-out local policy is the staged `full_v11` policy that lets `SETL` choose between:
  - immediate execution
  - `5m` confirmation
  - `10m` confirmation

### Important Honesty About V11

These `v11` numbers are promising, but they are **not yet final market-truth claims**.

Why:

- the backtest is archive-based, not a raw-bar end-to-end live simulation replay
- the cost model is still simplified / synthetic
- `PCOP` uses actual revealed `5m` / `10m` bars, which is valid for delayed entry logic, but still evaluated inside the branch-archive framework
- `SETL`, `PCOP`, `CESM`, and `PMWM` are trained and validated on the regenerated `v10` archive features, not yet on a broader raw execution log

So the current correct phrasing is:

- `v11` strongly improves the local research backtest
- `v11` plausibly bridges the execution gap
- `v11` still needs stricter end-to-end walk-forward or live-style confirmation before any production-strength claim

Important local verification for this pass:

- `tests.test_v11_research`: passes
- `tests.test_walkforward`: passes
- `tests.test_v9_runtime`: passes

## V12 Execution-Consistency Pass

V12 is now the active layer on top of the existing `v10` and `v11` stack.

Core purpose:

- close the archive-vs-execution gap that broke `v11` in Backtrader
- make feature computation causal and bar-consistent
- replace fragile absolute execution scoring with a contrastive ranker
- enforce a staged validation protocol before trusting any live claim

### What Landed

Implemented in-code:

- `src/v12/bar_consistent_features.py`
- `src/v12/feature_consistency_audit.py`
- `src/v12/tctl.py`
- `src/v12/sarv.py`
- `src/v12/live_confidence_calibrator.py`
- `src/v12/wfri.py`
- `src/v12/crowd_emotional_momentum.py`
- `src/v12/backtrader_strategy.py`

Runnable scripts:

- `scripts/run_feature_consistency_audit.py`
- `scripts/train_v12_tctl.py`
- `scripts/run_sarv_validation.py`
- `scripts/run_v12_backtrader_month.py`
- `scripts/run_v12_backtrader_walk_forward.py`
- `scripts/build_v12_summary.py`

New final reports:

- `outputs/v12/feature_consistency_report.json`
- `outputs/v12/tctl_evaluation_report.json`
- `outputs/v12/sarv_report.json`
- `outputs/v12/backtrader_month_2023_12.json`
- `outputs/evaluation/v12_summary.json`
- `outputs/evaluation/v12_summary.md`

### Important V12 Systems Fix

The original `scripts/run_feature_consistency_audit.py` looked hung because the older online replay path was recomputing the full feature buffer bar by bar. That has now been fixed.

Current V12 audit behavior:

- prints progress immediately
- uses the canonical causal BCFE path
- completes in seconds instead of appearing dead for many minutes

### Phase 1 Audit Result

Feature consistency on the current local stack:

- pass count: `25`
- fail count: `11`
- BCFE self-check fail count: `0`

Current failing features:

- `ema_cross`
- `dist_to_high`
- `dist_to_low`
- `hh`
- `ll`
- `volume_ratio`
- `session_asian`
- `session_london`
- `session_ny`
- `dow_sin`
- `dow_cos`

Current V12 rule:

- features that failed the audit are not treated as production-trustworthy V12 inputs
- the BCFE-consistent passed subset is the safe path

### TCTL Result

Latest local `v12` TCTL training:

- feature count: `50`
- train candidates: `19,200`
- valid candidates: `4,800`
- held-out pairwise accuracy: `0.484131`
- score diversity on valid set: `2,393` distinct rounded values
- device: `cuda`

Important interpretation:

- this is better than the `v11` constant-score collapse pattern
- but the ranker is still weak on held-out ordering quality
- V12 solved transfer-consistency more clearly than raw ranking power

### SARV Result

Latest SARV report on the most recent `90`-day replay window:

- Stage 1 win rate: `0.597222`
- Stage 2 win rate: `0.597222`
- Stage 2 gap: `0.0`
- Stage 1 participation: `0.533333`
- Stage 2 participation: `0.533333`
- Stage 3: pending

Important honesty:

- V12 currently clears the primary gap metric on this local replay slice
- but SARV still does **not** fully pass because:
  - Stage 1 participation is above the allowed band
  - Stage 3 paper-trade validation does not exist yet

### One-Month Backtrader Result

The required one-month Backtrader run was executed on `2023-12`.

Why not `2024-01`:

- current `v10` branch-feature coverage ends at `2023-12-29`
- so a true `2024-01` V12 month cannot be built from the current archived candidate set

Latest local month result:

- month: `2023-12`
- starting capital: `$1000`
- final capital: `$1000`
- return: `0.0%`
- trades executed: `0`
- plans generated: `1`
- replay calibration error: `0.49805`

Important interpretation:

- this is materially better than the catastrophic `v11` Backtrader collapse in the narrow sense that the execution pipeline no longer explodes from feature mismatch
- but it is **not yet a usable trading result**
- the current calibrated gate plus WFRI policy is too conservative and produces effectively no month activity

### WFRI / Calibration Status

Current deployment profile from the local month run:

- deployable regime classes: `trending_up` only
- calibration error remains far above the `sub-5%` V12 aspiration

So the current honest V12 read is:

- BCFE is real
- the audit infrastructure is real
- the archive-vs-replay gap appears much tighter
- TCTL no longer collapses to one score
- but the current TCTL + calibration + regime gating stack is still not production-ready

### Current V13 Recommendation

Next strongest move:

- keep BCFE as the only canonical feature path
- improve TCTL pair construction and threshold selection
- fix the overly conservative confidence calibration path
- run a real Stage 3 paper-trade window before any live deployment claim

## V13 Execution-Layer Research Status

### V13 Goal

`V13` was the execution-layer repair cycle after `V12` proved that feature consistency was much better but the trade stack was still too weak and too conservative.

The full local V13 prompt is now completed end-to-end.

### What Was Implemented

New local V13 architecture landed across:

- `src/v13/cabr.py`
- `src/v13/rcpc.py`
- `src/v13/uts.py`
- `src/v13/s3pta.py`
- `src/v13/daps.py`
- `src/v13/mbeg.py`
- `src/v13/lrtd.py`
- `src/v13/policy_utils.py`

And the supporting execution / evaluation path was wired through:

- `src/v12/backtrader_strategy.py`
- `src/v12/sarv.py`
- `src/service/app.py`
- `scripts/train_v13_cabr.py`
- `scripts/evaluate_v13_cabr.py`
- `scripts/run_s3pta.py`
- `scripts/run_v12_backtrader_month.py`
- `scripts/run_v12_backtrader_walk_forward.py`

### Phase 0 Truth

V13 continues to trust only the `25` BCFE-consistent features from V12 and rejects the `11` failing features.

### Phase 1 and Phase 2 Result

Patched regime-fixed V12 baseline:

- held-out pairwise accuracy: `0.499369`

CABR result:

- held-out pairwise accuracy: `0.531314`
- delta vs patched V12 baseline: `+0.031945`
- delta vs original V12 summary value: `+0.047183`

Honest interpretation:

- CABR is a real step forward
- but it still missed the `> 0.56` target from the V13 prompt

### RCPC / UTS / S3PTA / DAPS / MBEG / LRTD Result

Current local status:

- RCPC switched to learned calibration after `54` real paper trades
- current RCPC calibration error: `0.391655`
- S3PTA paper trades: `54`
- S3PTA paper-trade win rate: `0.388889`
- UTS deployable regimes in the current month run: `panic_shock`, `ranging`, `trending_down`, `trending_up`
- DAPS average lot in the month replay: `0.018824`
- MBEG veto rate in the month replay: `0.0`
- LRTD suppression rate in the month replay: `0.119048`

### December 2023 Backtrader Month

Latest required V13 month replay:

- month: `2023-12`
- starting capital: `$1000`
- final capital: `$1015.49`
- net profit: `$15.49`
- return: `+1.549369%`
- trades executed: `17`
- win rate: `0.588235`
- max drawdown: `1.110708%`
- profit factor: `1.740793`
- Stage 1 vs Stage 2 gap: `0.000005`

Important comparison versus V12:

- V12 month replay: `$1000 -> $1000`, `0` trades
- V13 month replay: `$1000 -> $1015.49`, `17` trades

So the practical V13 improvement is real: the system moved from no usable month activity to a passing month replay under the required Backtrader path.

### V13 Walk-Forward Result

Full walk-forward replay across all available replay months:

- months replayed: `37`
- aggregate trades: `570`
- aggregate win rate: `0.675439`
- aggregate return sum: `89.774457%`
- profitable months: `32 / 37`
- objective-pass months: `21 / 37`
- max single-month drawdown: `11.463713%`
- average monthly return: `2.426337%`
- average monthly trades: `15.405405`

### Current Honest V13 Read

V13 is the first local version that looks meaningfully viable in realistic replay terms.

What is now true:

- the full V13 prompt was completed
- CABR improved ranking quality materially
- RCPC calibration state now actually progresses from priors to learned calibration
- the V13 Backtrader month objective was met
- the walk-forward profile is directionally strong

What is still not solved:

- CABR still missed the prompt target of `> 0.56`
- RCPC calibration error is still far too high for a confident production claim
- S3PTA paper-trade quality is still not strong enough to call execution fully solved
- only `21` of `37` walk-forward months met the full objective band

### V14 Recommendation

Keep the BCFE plus CABR plus UTS stack, improve CABR beyond `0.56` with stronger context conditioning, materially reduce RCPC calibration error, and extend Stage 3 paper-trade validation before any live deployment claim.

## V14 Research Systems Status

### V14 Goal

`V14` was the research cycle focused on three genuinely new systems plus two execution-layer upgrades:

- `ACM` (`Asymmetric Crowd Memory`)
- `BST` (`Branch Survival Testing`)
- `SSC` (`Simulation Self-Critique`)
- temporal `CABR`
- `RSC` (`Regime-Stratified Calibration`)
- `LDRG` (`Live Deployment Readiness Gate`)

The local V14 prompt has now been completed end-to-end.

### What Was Implemented

New V14 modules landed in:

- `src/v14/acm.py`
- `src/v14/bst.py`
- `src/v14/ssc.py`
- `src/v14/rsc.py`
- `src/v14/ldrg.py`
- `src/v14/policy_utils.py`

Supporting scripts and artifacts landed in:

- `scripts/rebuild_branch_archive_v14.py`
- `scripts/train_v14_ssc.py`
- `scripts/check_ldrg_status.py`
- `scripts/run_v14_backtrader_month_internal.py`
- `scripts/run_v14_backtrader_walk_forward_internal.py`
- `scripts/build_v14_summary.py`

And V14 extended the existing stack in:

- `src/simulation/personas.py`
- `src/v13/cabr.py`
- `src/v13/uts.py`
- `scripts/train_v13_cabr.py`
- `scripts/run_v12_backtrader_month.py`
- `scripts/run_v12_backtrader_walk_forward.py`

### Phase 1 to Phase 5 Result

V14 archive rebuild and model training produced:

- V14 branch archive rows: `77817`
- V14 branch archive columns: `129`
- trusted BCFE feature count carried forward: `25`
- mean `bst_survival_score`: `0.797247`
- mean `fear_index_retail`: `0.006196`
- mean `fear_index_institutional`: `0.004462`

Temporal CABR result:

- held-out pairwise accuracy: `0.641945`
- V13 snapshot CABR baseline: `0.531314`
- delta vs V13: `+0.110631`
- target `> 0.56`: reached

Per-regime temporal CABR accuracy:

- `panic_shock`: `0.628223`
- `ranging`: `0.642303`
- `trending_down`: `0.679702`
- `trending_up`: `0.610354`

SSC result:

- assumption risk MAE: `0.173623`
- context consistency MAE: `0.467224`
- contradiction depth MAE: `0.112124`
- composite critique score mean: `0.791785`
- device: `cuda`

### Phase 8 Walk-Forward Result

Full V14 walk-forward replay across all available months:

- months replayed: `37`
- aggregate trades: `61`
- aggregate win rate: `0.770492`
- aggregate return sum: `17.134118%`
- aggregate profit factor: `4.242104`
- profitable months: `24 / 37`
- objective-pass months: `0 / 37`
- max single-month drawdown: `1.405340%`
- average Stage 1 vs Stage 2 gap: `0.000576`
- average SSC rejection rate: `0.0`
- average BST survival score: `0.852102`

RSC walk-forward status:

- paper trades accumulated: `61`
- paper-trade win rate: `0.786885`
- learned regimes: `ranging`
- max RSC calibration error: `0.513619`
- per-regime RSC error:
  - `ranging`: `0.367899`
  - `panic_shock`: `0.513619`
  - `trending_down`: `0.400676`

### December 2023 Backtrader Month

Latest required V14 month replay:

- month: `2023-12`
- starting capital: `$1000`
- final capital: `$1010.04`
- net profit: `$10.04`
- return: `+1.003767%`
- trades executed: `3`
- win rate: `0.666667`
- max drawdown: `1.405340%`
- profit factor: `1.697216`
- Stage 1 vs Stage 2 gap: `0.001171`

Comparison versus V13:

- V13 month replay: `$1015.49`, `17` trades, `0.588235` win rate, `1.740793` profit factor
- V14 month replay: `$1010.04`, `3` trades, `0.666667` win rate, `1.697216` profit factor

So V14 improved ranking quality and selectivity, but the practical month profile became too conservative.

### LDRG Result

Current local `LDRG` status:

- tier: `0`
- recommendation: `Continue research. Tier 1 not yet complete.`
- blocking criteria:
  - `wf_profitable_months_85pct`
  - `s3pta_200plus_trades`
  - `rsc_error_below_020`

### Current Honest V14 Read

V14 is a meaningful research improvement, but it is not a deployment-readiness pass.

What is now true:

- the full V14 prompt was completed
- temporal CABR materially exceeded the `> 0.56` target
- BST appears directionally helpful for robustness, with higher-BST months showing better average profit factor than lower-BST months
- walk-forward drawdown improved sharply versus V13
- LDRG is now formalised and the remaining blockers are explicit

What is still not solved:

- RSC calibration error is still far above the `0.20` target
- the V14 policy under-trades badly relative to the required month objective
- SSC is trained and integrated, but it is not yet affecting live decisions in practice because rejection rate stayed at `0.0`
- profitable month breadth fell below the Tier 1 requirement
- V14 still does not justify live deployment

### V15 Recommendation

V15 should keep the temporal CABR gains, keep ACM and BST, and focus on execution-policy recovery rather than another ranking jump.

Strongest next moves:

- reduce excessive conservatism in `LRTD` and thresholding so trade coverage recovers
- push `RSC` below `0.20` per regime with more paper trades and stronger regime-specific calibration constraints
- make `SSC` operationally meaningful as a real veto or size-scaling signal
- run an explicit `BST` ablation so the robustness contribution is measured causally, not only observationally

### V14 Summary Artifacts

Final V14 deliverables were written to:

- `outputs/evaluation/v14_summary.json`
- `outputs/evaluation/v14_summary.md`
- `outputs/v14/research_paper_outline.md`
- `outputs/v14/ldrg_status.json`

## V15 Predictability Recovery Status

### V15 Goal

`V15` is the execution-recovery cycle focused on recovering participation without throwing away the temporal CABR gains from `V14`.

Primary target:

- recover participation toward `15-50` trades per month
- keep win rate above `0.58`
- keep profit factor above `2.5`

New V15 systems planned and now at least partially implemented locally:

- `PRP` (`Participation Recovery Probe` / participation audit)
- `CPM` (`Conditional Predictability Mapper`)
- `PCE` (`Predictability-Conditioned Execution`)
- `CBWF` (`Calibration Bootstrap From Walk-Forward`)
- `ECI` (`Economic Calendar Integration`)
- `PSMP` (`Practical / minimum viable position sizing`)

### What Was Implemented

New V15 modules landed in:

- `src/v15/participation_audit.py`
- `src/v15/cpm.py`
- `src/v15/pce.py`
- `src/v15/cbwf.py`
- `src/v15/eci.py`
- `src/v15/policy_utils.py`

Supporting scripts and assets landed in:

- `scripts/run_participation_audit.py`
- `scripts/build_cpm_labels.py`
- `scripts/bootstrap_rsc_from_walkforward.py`
- `scripts/build_eci_historical.py`
- `scripts/run_v15_backtrader_month_internal.py`
- `scripts/run_v15_backtrader_walk_forward_internal.py`
- `data/economic_calendar/README.md`

And V15 extended the existing execution stack in:

- `src/v13/daps.py`
- `src/v12/backtrader_strategy.py`
- `scripts/run_v12_backtrader_month.py`
- `scripts/run_v12_backtrader_walk_forward.py`
- `config/project_config.py`

Important sizing and leverage note:

- lot sizing is now explicitly proportional to account equity, so growth from `$1000` to `$1100+` can increase allowed lot size
- a `max_account_leverage` control was added and defaults to `1:200` in the V15 runner path
- but the actual V15 research replay was still kept at broker leverage `1.0` because the V15 directive explicitly said not to use leverage in the Backtrader validation run

### Phase 0 Participation Audit Result

The V14 walk-forward participation audit is now explicit instead of inferred.

Audit result from `outputs/v15/participation_audit_v14.json`:

- total candidate bars inspected: `1600`
- final trades: `61`
- final trade rate: `0.038125`
- zero-trade months: `5`
  - `2021-10`
  - `2022-02`
  - `2022-09`
  - `2023-07`
  - `2023-08`

Primary bottleneck:

- gate: `lrtd_stability`
- blocked count: `1355`
- pass rate: `0.119558`

Secondary bottleneck:

- `uts_threshold`
- blocked count: `184`

Important interpretation:

- the V14 participation collapse was mainly caused by `LRTD`, not by `SSC`, not by `MBEG`, and not by DAPS lot size
- this validates the V15 theory direction: the previous gate stack was over-suppressing before trade quality even got a chance to act

### CPM Build Result

The full historical CPM label build now exists locally.

From `outputs/v15/cpm_summary.json`:

- labelled rows: `6024602`
- mean predictability: `0.497008`
- share above `0.60`: `0.466111`
- share above `0.70`: `0.149604`
- share above `0.80`: `0.091289`

Important interpretation:

- predictability is not uniformly absent, but truly high-predictability bars are materially sparser than medium-predictability bars
- this supports the idea that the system should deploy conditionally, not continuously

### CBWF Bootstrap Result

The V15 calibration bootstrap now loads historical walk-forward trade logs from V13 and V14 and seeds regime-specific RSC state.

Bootstrap result from `checkpoints/v15/rsc_bootstrapped.pkl`:

- total trades bootstrapped: `631`
- learned regimes:
  - `panic_shock`
  - `ranging`
  - `trending_down`
  - `trending_up`

Per-regime calibration error after bootstrap:

- `ranging`: `0.456990`
- `panic_shock`: `0.355881`
- `trending_down`: `0.124477`
- `trending_up`: `0.442308`

Max calibration error:

- `0.456990`

Important interpretation:

- `CBWF` materially improves the amount of calibration evidence available
- but it does not solve calibration on its own
- `trending_down` is acceptable
- `ranging`, `panic_shock`, and `trending_up` are still far above the `0.20` target

### ECI Status

The ECI scaffolding is implemented, but the repo does not yet contain a normalized historical event CSV covering the required macro releases.

Current practical state:

- `src/v15/eci.py` is implemented
- `scripts/build_eci_historical.py` is implemented
- `data/economic_calendar/README.md` documents the expected input format
- current V15 replay therefore ran with neutral / empty ECI context rather than a populated event calendar

### December 2023 V15 Pilot Month

The first V15 month replay was run on `2023-12`.

Initial V15 month result:

- month: `2023-12`
- starting capital: `$1000`
- final capital: `$1000.00`
- net profit: `$0.00`
- return: `0.0%`
- trades executed: `0`
- win rate: `0.0`
- max drawdown: `0.0%`
- profit factor: `null`
- Stage 1 vs Stage 2 gap: `0.000479`
- PCE threshold: `0.666667`
- skip reason breakdown:
  - `pce_not_predictable`: `42`

The V15 prompt explicitly required that if trades were too low, one threshold should be adjusted and the month rerun. That was done.

Adjustment 1:

- changed `pce_target_rate` from `0.20` to `0.45`
- tuned PCE threshold moved from `0.666667` to `0.583333`
- result: still `0` trades

Adjustment 2:

- changed `pce_target_rate` from `0.45` to `0.60`
- tuned PCE threshold moved from `0.583333` to `0.500000`
- result: still `0` trades

Post-adjustment blocking reasons on the same month:

- `low_agreement_0.000`: `19`
- `low_agreement_0.333`: `8`
- `regime_unstable_1bars`: `7`
- `regime_unstable_2bars`: `3`
- `regime_unstable_6bars`: `2`
- `regime_unstable_7bars`: `2`
- `atr_outside_range_13pct`: `1`

Important interpretation:

- after removing the very high CPM cutoff, the remaining blockers are mostly `agreement` and `regime stability`
- this means the current V15 pilot is still over-abstaining, just in a more transparent and principled way than V14
- the implementation is in place, but the policy is not yet tuned enough to recover participation

### Current Honest V15 Read

V15 is now structurally implemented, but not yet validated as a successful research pass.

What is now true:

- the V15 participation problem is much better diagnosed than before
- the previous dominant blocker is now documented explicitly
- CPM labels exist across the historical archive
- PCE, CBWF, ECI scaffolding, and PSMP sizing are now wired into the local codebase
- lot sizing now scales with equity and supports an optional `1:200` leverage ceiling for future execution modes
- V15 utilities and focused tests run successfully locally

What is still not solved:

- the December 2023 V15 pilot still executed `0` trades
- participation recovery has therefore not yet been achieved
- full V15 walk-forward was not run yet because the month pilot is still failing at the participation stage
- `CBWF` did not reduce max calibration error below `0.20`
- `ECI` is implemented in code but still lacks populated historical event inputs

Bottom line:

- V15 is a completed implementation pass, not yet a completed research-success pass
- the codebase is further along
- the policy calibration is still not there

### V16 Recommendation

The next cycle should not add more model complexity first. It should finish the participation recovery loop already started in V15.

Strongest next moves:

- relax `PCE` agreement and regime-stability requirements carefully, because those are now the dominant blockers after threshold loosening
- populate real historical `ECI` event data so predictable post-release windows can actually contribute
- rerun the one-month pilot until the system gets back into the `15-50` trade band before attempting full walk-forward
- only after month-level participation recovers should full V15-style walk-forward be run and summary artifacts be promoted

### V15 Working Artifacts

Current V15 working artifacts now present locally:

- `outputs/v15/participation_audit_v14.json`
- `outputs/v15/cpm_labels.parquet`
- `outputs/v15/cpm_summary.json`
- `outputs/v15/backtrader_month_2023_12_v15.json`
- `checkpoints/v15/rsc_bootstrapped.pkl`
- `checkpoints/v15/rsc_runtime.pkl`

## V16 Always-On Simulator Status

### V16 Goal

V16 changes the product framing from a gate-heavy execution bot into an always-on simulator.

The practical focus of this pass was:

- keep the primary trading horizon centered on `15m`
- keep the simulator refresh cadence at `5m`
- remove precondition silence from the live display path
- add a hosted paper-trading surface with leverage, history, and PnL
- add optional NVIDIA NIM routing for Kimi or Qwen through the existing OpenAI-compatible sidecar path

### Phase 0 Verification

On `2026-04-06`, the required V15 artifacts were re-verified locally and all were present:

- `src/v15/cpm.py`
- `src/v15/participation_audit.py`
- `src/v14/acm.py`
- `src/v14/bst.py`
- `src/v13/cabr.py`
- `src/v12/bar_consistent_features.py`
- `src/v12/backtrader_strategy.py`
- `outputs/v15/cpm_labels.parquet`
- `checkpoints/v14/cabr_temporal.pt`
- `checkpoints/v15/rsc_bootstrapped.pkl`

Current gate reality before V16:

- legacy V15 research execution still contains blocking reasons such as `pce_not_predictable`, `minority_veto`, and `lot_below_minimum`
- the participation audit vocabulary still includes `wfri_not_deployable`, `lrtd_suppressed`, and `uts_below_threshold`
- the new V16 live simulator path does not use those as precondition display gates

### V16 Implementation Landed

New V16 modules added:

- `src/v16/confidence_tier.py`
- `src/v16/sqt.py`
- `src/v16/sel.py`
- `src/v16/csl.py`
- `src/v16/paper.py`

Core service and UI updates:

- `src/service/llm_sidecar.py`
- `src/service/live_data.py`
- `src/service/app.py`
- `src/ui/web.py`

What these changes do:

- convert the current live branch payload into a V16 simulation result with direction, cone, minority path, CABR proxy, BST survival score, CPM display score, confidence tier, SQT label, and suggested lot size
- keep the main live presentation focused on the `15m` path while still refreshing every `5m`
- expose Frequency Mode and Precision Mode through the live API and website
- add a paper-trading engine with:
  balance
  equity
  realized and unrealized PnL
  open positions
  closed trade history
  leverage selection up to `1:200`
- make paper-trade lot size scale with current equity, so growth from `$1000` to `$1100` can increase size automatically

### NVIDIA NIM Support

NVIDIA NIM support was added as an optional LLM route.

Configuration now supports:

- provider: `nvidia_nim`
- API key env var: `NVIDIA_NIM_API_KEY`
- base URL env var: `NEXUS_NVIDIA_NIM_BASE_URL`
- model env var: `NEXUS_NVIDIA_NIM_MODEL`
- live UI model override field for Kimi or Qwen model ids

Important implementation choice:

- no NVIDIA key was written into the repository, the journal, or any generated artifact

### Local Hosting Status

The V16 app was launched locally and is now hosted at:

- UI: `http://127.0.0.1:8016/ui`
- health: `http://127.0.0.1:8016/health`

Runtime logs:

- `outputs/v16/server_8016.out.log`
- `outputs/v16/server_8016.err.log`

### Local Verification

Verification completed locally:

- `C:\\Users\\rfsga\\miniconda3\\python.exe -m py_compile config\\project_config.py src\\service\\llm_sidecar.py src\\service\\live_data.py src\\service\\app.py src\\ui\\web.py src\\v16\\__init__.py src\\v16\\confidence_tier.py src\\v16\\sqt.py src\\v16\\sel.py src\\v16\\csl.py src\\v16\\paper.py tests\\test_v16_confidence_tier.py tests\\test_v16_sqt.py tests\\test_v16_sel.py tests\\test_v16_csl.py`
- `C:\\Users\\rfsga\\miniconda3\\python.exe -m unittest tests.test_v16_confidence_tier tests.test_v16_sqt tests.test_v16_sel tests.test_v16_csl`
- `http://127.0.0.1:8016/health`
- `http://127.0.0.1:8016/ui`
- `http://127.0.0.1:8016/api/paper/state?symbol=XAUUSD`
- `http://127.0.0.1:8016/api/simulate-live?symbol=XAUUSD&mode=frequency&llm_provider=lm_studio`
- live paper-trade API smoke: open then close one `XAUUSD` paper trade successfully at `0.05` lot

Live smoke response from the hosted V16 API:

- symbol: `XAUUSD`
- direction: `BUY`
- confidence tier: `low`
- SQT label: `HOT`
- ECI note: `No high-impact event pressure near the current 15m horizon.`
- suggested lot: `0.05`
- provider: `lm_studio`

### Honest V16 Read

What is now genuinely true:

- the project now has a hosted V16 live simulator UI focused on `15m`
- the app exposes a working paper-trading loop with leverage and trade history
- the sidecar stack can now route through NVIDIA NIM safely without storing the key in code
- V16 no longer treats the live display path as a trade/no-trade gate cascade

What is not yet complete:

- this pass did not complete the full V16 Backtrader month run
- this pass did not complete the full V16 walk-forward research run
- cone hit rate, direction hit rate, and minority rescue rate are therefore only live-runtime values right now, not final research metrics

Bottom line:

- V16 is now live as a local simulator-and-paper-trading product pass
- V16 is not yet closed as a full research-evaluation pass

### V16 Summary Artifacts

- `outputs/evaluation/v16_summary.json`
- `outputs/evaluation/v16_summary.md`

### V17 Recommendation

The next pass should now split clearly into two tracks instead of mixing them:

- use the hosted V16 UI for real paper-trading accumulation and operator feedback
- run dedicated V16 Frequency and Precision Backtrader / walk-forward research jobs separately, so the paper-trading product iteration does not get blocked by longer evaluation cycles

## V17 Biological-Memory + Relativistic-Cone Status

### V17 Goal

V17 extends the hosted `15m` product track with the new theoretical layer requested in the V17 prompt:

- Winner-Loser Testosterone Cycle (`WLTC`) state features for crowd behavior
- Multifractal Market Memory (`MMM`) features for persistence / anti-persistence structure
- Lee-style chaotic activation (`LeeCOC`) support inside the CABR stack
- Relativistic Cone boundaries added to the existing live path visualization
- a restored glassmorphism + neomorphism UI instead of replacing the visual identity with a flat dashboard
- explicit numeric explainability in the UI
- full packet logging of what is sent to Kimi every `15m`, including the raw context and a glossary for the numeric values

### Phase 0 Verification

On `2026-04-06`, before any V17 edits were applied, the pre-existing V16 product server was confirmed to still be running locally:

- V16 UI: `http://127.0.0.1:8016/ui`
- V16 health: `http://127.0.0.1:8016/health`

The V17 pass therefore continued as a separate local host instead of replacing the V16 runtime already in service.

### V17 Implementation Landed

New V17 modules added:

- `src/v17/wltc.py`
- `src/v17/mmm.py`
- `src/v17/lee_coc.py`
- `src/v17/relativistic_cone.py`

Additional scripts and tests added:

- `scripts/build_mmm_features.py`
- `scripts/run_v16_backtrader_walk_forward.py`
- `tests/test_v17_wltc.py`
- `tests/test_v17_mmm.py`
- `tests/test_v17_lee_coc.py`
- `tests/test_v17_relativistic_cone.py`

Core integration files updated:

- `src/simulation/personas.py`
- `src/service/live_data.py`
- `src/service/llm_sidecar.py`
- `src/service/app.py`
- `src/v13/cabr.py`
- `src/v16/csl.py`
- `scripts/train_v13_cabr.py`
- `src/ui/web.py`

What these changes now do:

- build `WLTC` state features from recent bars and feed them into the live context
- compute live and historical `MMM` features, including `hurst_overall`, `hurst_positive`, `hurst_negative`, and `hurst_asymmetry`
- allow CABR training to use `MMM` context and optional `LeeCOC` activation for ablation testing
- replace the old single cone boundary with a relativistic cone structure containing inner band, outer hard boundary, and minority path plausibility
- restore the website to a glassmorphism + neomorphism presentation while keeping the live trading panels and paper-trading controls
- add an in-UI `What The Numbers Mean` section so the main displayed numerical values are explained directly in the product
- log each Kimi / NVIDIA NIM request packet in `15m` buckets with:
  exact system prompt
  exact user prompt
  raw structured context
  numeric glossary describing the numeric fields
  provider / model metadata
  success or error status

### V17 UI and Explainability

The V17 UI was rebuilt specifically to recover the requested visual theme and to make the model state easier to inspect.

What the hosted V17 UI now includes:

- glassmorphism + neomorphism cards instead of the flatter replacement style from the interrupted pass
- `15m` hero metrics and confidence summary
- relativistic cone chart with:
  inner cone
  dashed outer boundary
  minority branch
- paper-trading state with leverage, equity, open positions, closed history, and realized / unrealized PnL
- recent simulation panels
- judge / news panels
- a `What The Numbers Mean` section that explains the displayed numerical metrics
- a `Kimi 15m Packet` section that shows the logged prompt payload, raw context, and numeric glossary

### Kimi / NVIDIA NIM Logging

V17 now records the exact data package sent to Kimi through the NVIDIA NIM route every time the sidecar is called.

Important implementation details:

- packet logs are stored at `outputs/v17/kimi_packet_log.jsonl`
- packets are bucketed by `15m` UTC interval
- the live API now exposes `GET /api/llm/kimi-log`
- the hosted V17 website renders the latest packet directly in the UI

What the logged packet contains:

- request kind such as `market_context`
- symbol
- provider
- model id
- base URL
- exact system prompt
- exact user prompt
- raw structured context payload
- numeric glossary for values such as:
  `hurst_overall`
  `hurst_positive`
  `hurst_negative`
  `hurst_asymmetry`
  `cone_width`
  `cone_c_m`
  `cone_h_plus`
  `cone_h_minus`
  `cabr_score`
  `bst_proxy`
  `cpm_display`
  `agreement`
  `testosterone_index`
  `fundamental_tracking`
  `suggested_lot`

Smoke-status note:

- the first direct NVIDIA NIM packet smoke used model `moonshotai/kimi-k2-instruct`
- the request returned `HTTP Error 404: Not Found`
- the packet was still logged successfully, so the observability path is working even when the remote call fails

### MMM Historical Build

Historical `MMM` features were generated and written to:

- `outputs/v17/mmm_features.parquet`
- `outputs/v17/mmm_summary.json`

Current summary values:

- rows: `8000`
- timestamp min: `2009-03-15T20:42:00+00:00`
- timestamp max: `2023-12-29T14:48:00+00:00`
- mean `hurst_overall`: `1.100902`
- mean `hurst_positive`: `1.092719`
- mean `hurst_negative`: `1.093658`
- mean `hurst_asymmetry`: `-0.000941`

### Deferred V16 Walk-Forward Proxy Results

The V16 research track was finally completed in proxy form over the historical candidate archive so the product track and research track are now both documented.

Artifacts:

- `outputs/v16/backtrader_walkforward_frequency.json`
- `outputs/v16/backtrader_walkforward_precision.json`

Frequency Mode aggregate results:

- trades: `295`
- win rate: `0.908475`
- profit factor: `23.999570`
- cone hit rate: `0.644068`

Precision Mode aggregate results:

- trades: `60`
- win rate: `0.933333`
- profit factor: `74.158842`
- cone hit rate: `0.933333`

Important interpretation:

- these proxy results are strong on selectivity and path quality
- they still miss the desired trade-count floor for a higher-frequency `15m` operating style

### V17 CABR Ablation Results

V17 also tested whether the new theoretical additions improved CABR temporal quality versus the V14 temporal baseline.

Artifacts:

- `outputs/v17/cabr_eval_mmm_only.json`
- `outputs/v17/cabr_eval_lee_only.json`
- `outputs/v17/cabr_eval_mmm_lee.json`
- `outputs/v17/cabr_evaluation_report_v17.json`

Temporal comparison:

- V14 temporal baseline: `0.641945`
- `MMM` only: `0.526448`
- `LeeCOC` only: `0.493486`
- `MMM + LeeCOC`: `0.503218`

Interpretation:

- the V17 CABR ablations did not beat the V14 temporal baseline
- the theory is now wired into the codebase, but the current training outcome is weaker than the best previous temporal checkpoint

### V17 Walk-Forward Proxy Results

V17 proxy walk-forward reports were generated using the V17 CABR variant.

Artifacts:

- `outputs/v17/backtrader_walkforward_frequency_v17.json`
- `outputs/v17/backtrader_walkforward_precision_v17.json`

Frequency Mode aggregate results:

- trades: `124`
- win rate: `0.943548`
- profit factor: `63.726146`
- cone hit rate: `0.701613`

Precision Mode aggregate results:

- trades: `36`
- win rate: `0.944444`
- profit factor: `59.168964`
- cone hit rate: `0.944444`

Interpretation:

- the V17 proxy layer stayed very selective and statistically strong on the executed subset
- participation fell even further than the deferred V16 frequency proxy
- the trade-count objective is still not met

### Local Hosting Status

The V17 app was launched as a separate local product host so the original V16 runtime remained intact.

Current local URLs:

- V17 UI: `http://127.0.0.1:8017/ui`
- V17 health: `http://127.0.0.1:8017/health`
- V17 Kimi log API: `http://127.0.0.1:8017/api/llm/kimi-log`

Current runtime logs:

- `outputs/v17/server_8017.out.log`
- `outputs/v17/server_8017.err.log`

Paper-trading state at the time of journal update:

- balance: `$8081.20`
- equity: `$8081.20`
- realized PnL: `$7081.20`
- total trades: `120`
- open positions: `0`
- win rate: `1.000000`

### Local Verification

Verification completed locally:

- `C:\\Users\\rfsga\\miniconda3\\python.exe -m py_compile config\\project_config.py src\\v17\\__init__.py src\\v17\\wltc.py src\\v17\\mmm.py src\\v17\\lee_coc.py src\\v17\\relativistic_cone.py src\\simulation\\personas.py src\\service\\llm_sidecar.py src\\service\\live_data.py src\\service\\app.py src\\ui\\web.py src\\v13\\cabr.py src\\v16\\csl.py scripts\\build_mmm_features.py scripts\\run_v16_backtrader_walk_forward.py scripts\\train_v13_cabr.py tests\\test_v17_wltc.py tests\\test_v17_mmm.py tests\\test_v17_lee_coc.py tests\\test_v17_relativistic_cone.py`
- `C:\\Users\\rfsga\\miniconda3\\python.exe -m unittest tests.test_v17_wltc tests.test_v17_mmm tests.test_v17_lee_coc tests.test_v17_relativistic_cone tests.test_v16_csl`
- `http://127.0.0.1:8017/health`
- `http://127.0.0.1:8017/ui`
- `http://127.0.0.1:8017/api/paper/state?symbol=XAUUSD`
- `http://127.0.0.1:8017/api/llm/kimi-log?limit=4`

Important runtime note:

- the hosted V17 UI and lightweight endpoints were verified successfully
- the full `/api/simulate-live` smoke from the shell timed out during the short smoke window, so that endpoint was not re-verified end-to-end in this journal pass

### Honest V17 Read

What is now genuinely true:

- V17 is hosted locally as a separate product pass at `8017`
- the requested glassmorphism + neomorphism UI style has been restored
- the product now exposes in-UI explanations for the major numerical values
- the Kimi packet logging and glossary path is implemented and visible in both the API and the website
- `WLTC`, `MMM`, and relativistic cone logic are now integrated into the live context and visualization path

What is not yet true:

- the current NVIDIA NIM smoke packet returned `404`, so the remote Kimi route still needs provider-level confirmation
- the V17 CABR ablations did not outperform the V14 temporal baseline
- both deferred V16 and V17 proxy walk-forward outputs still under-trade versus the desired high-frequency target
- the full live `/api/simulate-live` route was not re-verified within the smoke timeout window during this journal update

Bottom line:

- V17 is a real hosted product-and-observability pass
- V17 is not a research win over V14 yet
- the system is now easier to inspect, easier to operate, and much better instrumented for the next tuning cycle

### V17 Summary Artifacts

- `outputs/evaluation/v17_summary.json`
- `outputs/evaluation/v17_summary.md`

### V18 Recommendation

The next cycle should keep the new V17 observability stack, but treat the model work more ruthlessly:

- keep the V17 UI, packet logging, and relativistic cone instrumentation as the operator-facing base
- verify the exact NVIDIA NIM model id and request contract needed for Kimi before treating that route as production-ready
- do not promote `MMM` or `LeeCOC` into the main CABR checkpoint unless they beat the V14 temporal baseline in ablation
- focus the next research pass on recovering `15m` participation without destroying the strong win-rate / cone-hit characteristics already present in the selective proxy outputs
