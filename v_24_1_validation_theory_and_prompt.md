# V24.1 — Validation Phase Theory and Prompt

# Part I — Core Theory

## Why V24.1 Exists

V24 successfully implemented the architecture, but the project is still missing proof that the architecture produces a real trading edge.

The next phase is not about adding more modules.

It is about answering five questions:

1. Are the generated future branches realistic?
2. Does the diffusion model actually outperform the older generator?
3. Does the evolutionary population discover better strategies over time?
4. Does the system truly improve expectancy after abstention and risk filtering?
5. Does the system remain profitable across different regimes instead of only one lucky slice?

V24.1 therefore shifts the project from:

```text
implementation
    -> validation
```

---

# Core Hypothesis

The true edge does not come from predicting every candle.

The true edge comes from:

```text
finding rare situations where:
    branch agreement is high
    minority risk is low
    expected reward:risk is high
    historical analogs agree
    macro regime is supportive
```

Therefore the project should optimize:

```text
trade quality
```

not:

```text
raw directional accuracy
```

---

# V24.1 Research Priorities

## System A — Branch Realism Validation

The diffusion generator must now be judged like a scientific model.

For every generated branch:

Measure:
- path realism
- volatility realism
- drawdown realism
- regime realism
- structural realism

### Required Metrics

```text
branch realism score
= 0.25 * volatility realism
+ 0.20 * analog similarity
+ 0.20 * regime consistency
+ 0.20 * path plausibility
+ 0.15 * minority usefulness
```

### Validation Tests

For each historical market state:

1. generate 64 future branches
2. compare them against the true realized future
3. measure:
   - whether the real future lies inside the branch cloud
   - whether the minority branch captured the eventual move
   - whether the branch cone is too narrow or too wide

Important target:

```text
True future inside cone >= 70%
```

---

## System B — Diffusion vs Alternative Generators

The diffusion model must compete against:

- Conditional VAE
- Transformer decoder
- Mamba sequence model
- xLSTM sequence model

For each generator evaluate:

- branch realism score
- cone containment rate
- expected trade value after CABR selection
- runtime speed
- diversity of branches

The winner becomes the production generator.

Important theory:

The best model may not be the one with the best raw reconstruction loss.

The best model is the one whose generated branches lead to the strongest:

```text
post-selection trade expectancy
```

---

## System C — CABR Dangerous Branch Theory

The current CABR chooses the best branch.

V24.1 extends this.

Every market state should produce:

```text
best profitable branch
dangerous invalidation branch
```

Then compute:

```text
tradeability =
    best_branch_score
    - dangerous_branch_score
```

Only trade when:

```text
tradeability > threshold
```

This is more robust than confidence alone.

---

## System D — Evolutionary Agent Research

The evolutionary population must stop being static.

Instead of 10 fixed agents:

```text
agent
    -> evaluated weekly
    -> mutated
    -> replaced
```

### Fitness Function

```text
fitness =
    0.30 * expectancy
  + 0.20 * Sharpe
  + 0.15 * win rate
  - 0.20 * drawdown
  + 0.10 * stability
  + 0.05 * minority rescue rate
```

### Mutation Parameters

- confidence threshold
- tradeability threshold
- regime preference
- stop-loss multiplier
- take-profit multiplier
- diffusion branch weighting
- macro weighting
- sentiment weighting
- spread filter

Important theory:

The project should evolve:

```text
families of strategies
```

rather than search for one perfect strategy.

---

## System E — Calibration and Abstention

The current weakness is overconfidence.

V24.1 should explicitly train a calibration model.

Inputs:
- branch disagreement
- branch dispersion
- dangerous branch score
- analog agreement
- macro agreement
- recent system drawdown

Output:

```text
probability that this trade is genuinely worth taking
```

The system should abstain much more often.

Target:

```text
Participation 5%–20%
Win rate > 60%
Expected R multiple > 0.25
```

---

# Part II — V24.1 Prompt

```md
# Nexus Trader V24.1 — Validation And Selection Directive

You are the lead architect for Nexus Trader V24.1.

The V24 architecture is already implemented.

Your job is NOT to add more random modules.

Your job is to scientifically validate which parts of V24 actually create a real edge.

The project must now optimize for:

- trade quality
- calibrated abstention
- realistic future branch generation
- robust behavior across regimes

Never optimize only for raw BUY/SELL accuracy.

---

## Phase 0 — Validation Dataset

Create:

```text
outputs/v24_1/validation_dataset.parquet
```

For every historical market state include:

- world state features
- generated branches
- CABR scores
- dangerous branch score
- realized future path
- realized trade outcome
- macro regime
- branch realism metrics

---

## Phase 1 — Branch Realism Evaluation

Create:

```text
src/v24_1/branch_realism.py
scripts/evaluate_branch_realism.py
```

For each generated branch compute:

- volatility realism
- analog similarity
- regime consistency
- path plausibility
- minority usefulness

Then compute:

```text
branch_realism_score
```

Generate reports:

```text
outputs/v24_1/branch_realism_report.json
outputs/v24_1/branch_realism_report.md
```

Important metrics:

- cone containment rate
- minority rescue rate
- branch diversity
- realism score

---

## Phase 2 — Generator Tournament

Create:

```text
src/v24_1/generator_tournament.py
```

Compare:

- diffusion
- CVAE
- Transformer
- Mamba
- xLSTM

For every generator evaluate:

- branch realism score
- trade expectancy after CABR
- runtime
- cone containment

Save:

```text
outputs/v24_1/generator_leaderboard.json
```

Select the best generator automatically.

---

## Phase 3 — Dangerous Branch CABR

Upgrade CABR to emit:

```text
best_branch_score
dangerous_branch_score
tradeability_score
```

Create:

```text
src/v24_1/cabr_tradeability.py
```

Trade only when:

```text
tradeability_score > threshold
```

---

## Phase 4 — Evolutionary Agent Validation

Create:

```text
src/v24_1/evolution_runner.py
```

Run 10 generations.

Each generation:

- evaluate all agents
- remove bottom 30%
- copy top 20%
- mutate the survivors

Track:

- expectancy
- Sharpe
- drawdown
- minority rescue rate
- regime specialization

Save:

```text
outputs/v24_1/evolution_history.json
```

---

## Phase 5 — Calibration Model

Create:

```text
src/v24_1/calibration_model.py
```

Inputs:

- branch disagreement
- dangerous branch score
- analog agreement
- macro agreement
- recent performance

Output:

```text
true_trade_probability
```

Then use:

```text
true_trade_probability
```

instead of raw confidence.

---

## Phase 6 — Final Walk-Forward Validation

Run:

- 2023
- 2024
- 2025
- 2026

Measure:

- participation
- win rate
- expectancy
- Sharpe
- max drawdown
- cone containment
- minority rescue rate

Target:

```text
Participation: 5–20%
Win rate: >60%
Expectancy: >0.25R
Max DD: <20%
Cone containment: >70%
```

If those targets are not reached, do not proceed to V25.

Instead:

- identify which subsystem failed
- retrain only that subsystem
- rerun the tournament and validation cycle
```

