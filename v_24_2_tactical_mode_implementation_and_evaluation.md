# V24.2 — Tactical Mode Implementation, Evaluation, and Deployment Plan

# Executive Goal

V24.2 is not a new architecture.

V24.2 extends the validated V24.1 system with a second operating mode:

```text
Strategic Mode
    -> 15m / 30m
    -> low participation
    -> highest-quality trades

Tactical Mode
    -> 1m / 3m / 5m
    -> higher participation
    -> short-term entries aligned with the strategic bias
```

The purpose of V24.2 is:

- increase trade frequency without destroying expectancy
- keep the strategic engine as the higher-level supervisor
- only allow tactical trades when the larger market context supports them

---

# Core V24.2 Concept

The system becomes:

```text
Strategic Engine
    -> produces long-bias / short-bias / hold
    -> produces regime confidence
    -> produces dangerous-branch risk

Tactical Engine
    -> searches for short-term entries
    -> only trades in the same direction as the strategic engine
    -> uses faster branch generation and lighter filtering
```

Example:

```text
Strategic Engine:
    bullish next 30m
    regime confidence = 0.82
    dangerous branch risk = low

Tactical Engine:
    wait for 1m pullback
    wait for liquidity sweep below local low
    enter long when tactical branch reverses upward
```

---

# Tactical Mode Constraints

Tactical mode is only enabled if:

```text
regime_confidence > 0.75
tradeability_score > 0.40
true_trade_probability > 0.70
spread < tactical_spread_limit
```

Tactical mode must automatically disable when:

- spread widens suddenly
- major news event begins
- branch disagreement rises
- strategic engine switches direction
- recent tactical drawdown exceeds threshold

---

# V24.2 Implementation Phases

## Phase 0 — Tactical Dataset Creation

Create:

```text
outputs/v24_2/tactical_dataset.parquet
```

Include:

- 1m / 3m / 5m candles
- spread
- tick volume
- wick imbalance
- local liquidity sweep flags
- session type
- short-term volatility
- strategic engine outputs
- strategic branch direction
- strategic tradeability score
- realized short-term outcome

Target labels:

```text
entry_success_1m
entry_success_3m
entry_success_5m
```

Where success means:

```text
trade reaches +0.25R before -0.5R
```

---

## Phase 1 — Tactical Regime Detector

Create:

```text
src/v24_2/tactical_regime.py
```

Classify each short-term state as:

- trend continuation
- breakout
- mean reversion
- liquidity sweep reversal
- chop / no-trade

Required output:

```text
regime_type
regime_confidence
allow_tactical_trade
```

Target:

```text
No tactical trades during chop or uncertain regimes
```

---

## Phase 2 — Lightweight Tactical Generator

Create:

```text
src/v24_2/tactical_generator.py
```

Requirements:

- 16 branches instead of 64
- 1m / 3m / 5m horizon
- runtime under 500ms

Candidate generators:

- tiny diffusion model
- small Mamba model
- small Transformer

Expected output:

```text
16 short-term future paths
```

Each path should include:

- expected move
- invalidation level
- branch probability
- local volatility envelope

---

## Phase 3 — Tactical CABR

Create:

```text
src/v24_2/tactical_cabr.py
```

For each tactical state compute:

```text
best_short_branch
dangerous_short_branch
short_tradeability
```

Trade only if:

```text
short_tradeability > 0.35
```

And:

```text
short_tradeability aligns with strategic direction
```

---

## Phase 4 — Microstructure Layer

Create:

```text
src/v24_2/microstructure.py
```

Add:

- spread regime
- spread expansion rate
- wick imbalance
- liquidity sweep probability
- estimated slippage
- execution quality score

Execution quality:

```text
execution_quality =
    expected_profit
    - spread_cost
    - slippage_cost
```

Only trade if:

```text
execution_quality > 0
```

---

## Phase 5 — Tactical Calibration Model

Create:

```text
src/v24_2/tactical_calibration.py
```

Inputs:

- tactical tradeability
- spread
- liquidity sweep probability
- branch disagreement
- strategic confidence
- recent tactical win rate

Output:

```text
tactical_trade_probability
```

Trade only if:

```text
tactical_trade_probability > 0.75
```

---

## Phase 6 — Tactical + Strategic Integration

Create:

```text
src/v24_2/integrated_engine.py
```

Decision logic:

```text
if strategic engine says HOLD:
    no tactical trade

if tactical direction != strategic direction:
    reject tactical trade

if tactical probability high and aligned:
    allow tactical entry
```

The strategic engine becomes the supervisor.

The tactical engine becomes the entry optimizer.

---

# V24.2 Evaluation Plan

## Tactical Performance Metrics

Measure:

- tactical participation rate
- tactical win rate
- tactical expectancy
- tactical max drawdown
- tactical average holding time
- spread-adjusted expectancy

Target:

```text
Participation: 10–30%
Win Rate: >58%
Expectancy: >0.15R
Max Drawdown: <15%
Average Hold Time: 1–10 minutes
```

---

## Combined Strategic + Tactical Metrics

Measure:

```text
Strategic only
vs
Strategic + Tactical
```

Compare:

- total return
- drawdown
- Sharpe
- trade count
- expectancy
- recovery after losses

Target:

```text
Tactical mode should increase return
without materially increasing drawdown
```

---

## Stress Tests

Test V24.2 during:

- London open
- New York open
- CPI / FOMC days
- high-spread periods
- trend days
- range days

The tactical engine should automatically disable itself when:

```text
market quality too low
```

---

# Final V24.2 Deliverables

```text
outputs/v24_2/tactical_dataset.parquet
outputs/v24_2/tactical_generator_report.json
outputs/v24_2/tactical_tradeability_report.json
outputs/v24_2/tactical_calibration_report.json
outputs/v24_2/tactical_walkforward_report.json
outputs/v24_2/strategic_vs_tactical_report.json
```

---

# Final V24.2 Prompt

```md
# Nexus Trader V24.2 — Tactical Mode Directive

You are the lead engineer for Nexus Trader V24.2.

V24.1 already proved that the strategic architecture works.

Your task is NOT to replace the strategic engine.

Your task is to build a tactical mode that increases trade frequency while preserving the strategic edge.

Rules:

- Strategic engine remains the supervisor
- Tactical trades must align with strategic direction
- Tactical mode only activates in high-confidence regimes
- Tactical mode must disable automatically in poor market conditions

Required trade rule:

```text
Only enter tactical trade if:
    strategic_direction == tactical_direction
AND strategic_tradeability > 0.40
AND tactical_tradeability > 0.35
AND tactical_trade_probability > 0.75
AND execution_quality > 0
```

Targets:

```text
Tactical Participation: 10–30%
Tactical Win Rate: >58%
Tactical Expectancy: >0.15R
Combined System Drawdown: <20%
```

Do not optimize for maximum trade count.

Optimize for:

```text
more trades
without losing the strategic edge
```

