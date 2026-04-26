# V24 Architect.md

# Nexus Trader V24 — Unified Architecture

## Core Thesis

V24 abandons the idea that the market can be predicted with a single BUY / SELL / HOLD label.

Instead, V24 models:

```text
current market state
    -> distribution of plausible futures
    -> expected trade quality
    -> calibrated execution
```

The system no longer optimizes for direction alone.

It optimizes for:

- expected value
- reward:risk ratio
- downside danger
- robustness across regimes
- uncertainty calibration

---

# V24 Philosophy

V24 is built on five principles:

1. The future is a cloud, not a point.
2. Minority scenarios matter.
3. Trade quality matters more than direction.
4. Quantitative models should be learned, not manually weighted.
5. Multiple evolving agents outperform one fixed strategy.

---

# Full V24 Pipeline

```text
World State
    -> Quant Feature Layer
    -> Meta-Aggregator
    -> Conditional Diffusion Generator
    -> CABR V24 Branch Ranking
    -> Ensemble Risk Judge
    -> Evolutionary Agent Population
    -> OpenClaw Supervisor
    -> Human Approval / Execution
```

---

# 1. World State Layer

The world-state representation becomes the central input to every later module.

## Inputs

### Market Structure
- OHLCV
- multi-timeframe returns
- ATR
- realized volatility
- spread
- slippage estimate
- wick imbalance
- liquidity sweep detection
- session location
- distance from recent high/low

### Existing Nexus Features
- MMM multifractal memory
- WLTC cycle state
- MFG disagreement
- analog retrieval similarity
- CABR historical confidence
- branch disagreement statistics

### Quantitative Models
- Hidden Markov Model regime probabilities
- Kalman filter trend estimate
- stochastic volatility estimate
- GARCH volatility
- Bayesian filter state
- particle filter estimate
- Hurst exponent
- macro regime classifier

### Macro / Sentiment Inputs
- FinGPT sentiment
- FinNLP event extraction
- news severity
- macro calendar state
- geopolitical stress score
- social sentiment score

### Runtime State
- rolling win rate
- recent drawdown
- consecutive loss streak
- live performance trend
- recent direction bias
- recent agent performance

---

# 2. Learned Quant Fusion (Meta-Aggregator)

The previous versions used manually weighted rules.

V24 replaces this with a learned meta-model that decides which signals matter in which regime.

## Goal

Learn:

```text
(context -> trade quality)
```

instead of:

```text
(context -> BUY / SELL / HOLD)
```

## Architecture

```text
Inputs
    -> expert embeddings
    -> Transformer / xLSTM mixture-of-experts
    -> regime-aware gating network
    -> output heads
```

### Expert Heads
- trend expert
- reversal expert
- macro expert
- analog expert
- risk-control expert

### Outputs
- expected trade value
- probability of profitable trade
- expected reward:risk ratio
- expected drawdown
- uncertainty score
- danger score
- abstain probability

### Training Target

```text
expected_value
= probability(win) * expected_reward
- probability(loss) * expected_loss
```

Additional targets:
- realized R:R
- realized slippage
- realized volatility
- branch survival probability

---

# 3. Conditional Diffusion Future Generator

The old branch generator is replaced by a conditional diffusion model.

## Why Diffusion

Diffusion can model:
- heavy tails
- volatility clustering
- sudden regime shifts
- minority futures
- nonlinear path dependencies

## Generator Inputs
- current market state
- regime embedding
- analog memory
- macro state
- sentiment state
- microstructure state

## Generator Output

Generate 32–64 plausible future paths:

```text
future path 1
future path 2
future path 3
...
future path 64
```

Each path contains:
- OHLC
- volatility envelope
- drawdown path
- branch confidence

## Competing Generator Types

V24 should test:
- diffusion model
- conditional VAE
- autoregressive transformer
- Mamba-based world model

The production generator is whichever produces:
- best branch realism
- strongest backtest expectancy
- strongest minority-scenario quality

---

# 4. CABR V24 — Dual Branch Ranking

Previous CABR only tried to select the best branch.

V24 introduces two simultaneous scores:

```text
best_branch_score
dangerous_branch_score
```

## Best Branch
The branch most likely to produce a profitable trade.

## Dangerous Branch
The branch most likely to create a loss or invalidation.

## Uses
- stop placement
- position sizing
- abstention logic
- trade confidence

If:

```text
dangerous_branch_score > best_branch_score
```

then:

```text
abstain
```

---

# 5. Ensemble Risk Judge

The risk judge combines:
- V22 hybrid risk judge
- meta-aggregator output
- CABR danger branch
- live performance state
- HMM regime confidence

## Final Outputs
- execute
- abstain
- reduce size
- wait

## Risk Controls

Retain all V22 protections:
- circuit breaker
- regime persistence penalty
- maximum consecutive-loss protection
- minimum reward:risk threshold
- maximum spread threshold
- maximum slippage threshold

No trade is allowed unless:

```text
expected_RR >= 1.5
AND
uncertainty <= threshold
AND
danger_score <= threshold
```

---

# 6. Evolutionary Agent Population

V24 does not rely on one single strategy.

Instead, it runs a population of competing agents.

## Initial Agent Families

- trend-following
- mean-reversion
- macro-heavy
- analog-heavy
- high-abstention
- aggressive
- low-risk
- breakout specialist

Each agent has different:
- thresholds
- weighting
- confidence logic
- regime preference

## Evolution Cycle

Every week:

```text
bottom 30% removed
best 20% copied
new mutations created
```

### Mutation Examples
- confidence threshold
- regime sensitivity
- TP/SL multiplier
- danger weighting
- spread filter
- macro weighting

## Fitness Function

```text
fitness =
    0.35 * expectancy
  + 0.20 * Sharpe
  - 0.20 * drawdown
  + 0.15 * stability
  + 0.10 * trade_frequency
```

---

# 7. OpenClaw Supervisor

OpenClaw is not the trader.

It is the supervisor.

## OpenClaw Responsibilities
- monitor macro news
- monitor internet sentiment
- monitor geopolitical events
- monitor live system behavior
- select which agent family is active
- suggest mutations
- pause trading during major events

## OpenClaw Cannot
- directly place trades
- bypass the circuit breaker
- override human approval

---

# 8. Human Approval Layer

The final system should support three modes:

### Mode 1 — Research
No live execution.

### Mode 2 — Assisted
The system proposes trades and the human approves them.

### Mode 3 — Autonomous
The system may trade automatically only if:
- circuit breaker inactive
- confidence high
- danger score low
- recent drawdown acceptable

---

# 9. V24 Implementation Roadmap

## Phase 1 — Repair V22
- fix calibration
- remove SELL bias
- improve R:R filtering
- ensure trade frequency recovers

## Phase 2 — Meta-Aggregator
- train learned quant fusion model
- move from direction labels to trade-quality labels

## Phase 3 — Diffusion Generator
- train conditional diffusion branch generator
- compare against CVAE and Transformer

## Phase 4 — CABR V24
- add dangerous branch scoring
- integrate with risk judge

## Phase 5 — Agent Population
- create 5–10 evolving agents
- weekly mutation and selection

## Phase 6 — OpenClaw Integration
- add supervisory news and macro layer
- keep human approval mandatory

---

# 10. Expected End State

The final V24 system should behave like:

```text
A population of specialized trading minds
watching a cloud of possible futures
while continuously adapting to changing regimes
without becoming overconfident
```

The final design goal is not to predict the future perfectly.

The goal is:

```text
find trades where the upside is strong
and the dangerous futures are still acceptable
```

