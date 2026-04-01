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
- GPT-OSS is now integrated as a local sidecar
- manual technical-analysis panels now exist
- but the full MiroFish-like simulation logic is still not complete

Also important:

- current predictive quality is still early
- previous remote sample runs were only around low-50% classifier performance
- no honest claim of `90%+` predictive accuracy exists right now

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
  - minority guardrail score
- the UI now shows:
  - strongest surviving branch
  - minority invalidation branch
  - supporting branches
- GPT-OSS now acts as a bounded sidecar:
  - it tilts personas
  - it contributes a small numeric prior
  - it does not replace the simulator

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

1. Add historical analog branch scoring.
2. Add regime classification.
3. Improve reverse collapse and minority scenario logic further.
4. Improve confidence calibration from actual branch reliability.
5. Improve technical structure scoring.
6. Keep GPT-OSS as a structured sidecar.
7. Only then consider more advanced training loops.

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
- The project still needs deeper MiroFish-like branching, branch fitness scoring, reverse-confidence collapse, and better confidence calibration.

Highest-value next work:
1. Add historical analog scoring for branches.
2. Improve reverse collapse and minority scenario logic.
3. Deepen persona logic and regime handling.
4. Connect specialist bots more directly into branch scoring and multi-horizon consistency checks.
5. Improve technical market-structure logic.
6. Improve evaluation with walk-forward tests and cone hit-rate analysis.

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

The next serious step is not “bigger random model.”
The next serious step is:

- better historical branch scoring
- better collapse
- better confidence
- better bot-simulator integration
- better technical structure logic
- then stronger structured reasoning around it
