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

The key design idea is that uncertainty should be visible, not hidden.

## What The User Wants

The user wants a system much closer to the MiroFish-style idea:

- multiple trader personas acting at once
- many branches of possible futures
- a reverse-confidence collapse back to the root
- a probability cone that reflects disagreement
- a live UI showing real candles beside predicted future candles
- manual observation first, not automatic execution

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
- but the full MiroFish-like simulation logic is not yet realized

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

### Simulation Scaling Bug

There was a serious bug where branch probabilities were converted into prices using an unrealistic shortcut, causing absurd moves like gold jumping toward `4900` in 5 minutes.

That bug has been fixed.

Current cone logic now:

- aggregates actual branch price paths
- uses weighted mean branch prices
- uses weighted dispersion for cone width
- bounds confidence more realistically
- versions simulation history so stale broken runs are excluded

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
6. Reverse collapse should preserve every leaf as a vote.
7. Final confidence should come from:
   - branch directional agreement
   - branch price dispersion
   - branch fit quality
   - model calibration

That is much closer to the real target than the current lighter implementation.

## What To Improve Next

### 1. Branching Engine

Highest-value work:

- branch over simulated candle paths, not just simple repeated ABM steps
- store branch state per timestep
- compute branch fitness using historical analog windows
- keep more than one “winning” path alive
- collapse from all leaves, never a single winner

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

### 5. Evaluation

Current project still needs stronger evaluation than plain classifier metrics.

Priorities:

- walk-forward testing by year/month
- selective signal precision
- simulation hit-rate
- cone containment rate
- directional hit-rate after filtering
- regime-by-regime performance

### 6. UI / Monitoring

Need to keep improving:

- compare chart
- branch explorer
- minority scenario overlay
- branch-by-branch explanation
- source timestamps for every feed
- stale-feed warnings

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

1. Rebuild the branching engine around realistic branch price paths.
2. Add historical analog branch scoring.
3. Add regime classification.
4. Add confidence calibration from actual branch reliability.
5. Add minority-scenario tracking in the UI.
6. Add an LLM sidecar for macro/news/crowd interpretation.
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
- The project still needs deeper MiroFish-like branching, branch fitness scoring, reverse-confidence collapse, and better confidence calibration.

Highest-value next work:
1. Make branching more realistic and multi-step.
2. Add historical analog scoring for branches.
3. Improve reverse collapse and minority scenario logic.
4. Deepen persona logic and regime handling.
5. Add LLM sidecar for structured macro/news/crowd reasoning.
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
- preserved local model artifacts

But it is still not the full MiroFish-style simulator the user originally wanted.

The next serious step is not “bigger random model.”
The next serious step is:

- better branching
- better branch scoring
- better collapse
- better confidence
- then an LLM sidecar for structured reasoning

