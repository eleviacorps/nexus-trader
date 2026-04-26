# V24.1 Executive Summary, Achievements, Core Concepts, and Next Steps

# Executive Summary

V24 successfully completed the architectural phase of Nexus Trader.

The project now contains:

- world-state modeling
- branch generation
- CABR branch ranking
- diffusion / Mamba / xLSTM generator support
- evolutionary agent framework
- calibration framework
- OpenClaw supervision layer

However, V24 did not yet prove that those components create a real trading edge.

The strongest lesson from V24 is:

```text
The edge does not come from predicting every bar.

The edge comes from:
    finding rare, high-quality situations
    avoiding dangerous branches
    preserving minority scenarios
    abstaining when uncertainty is high
```

Therefore V24.1 shifts the project from:

```text
building more systems
    ->
scientifically validating which systems actually work
```

The next phase is not V25.

The next phase is proving:

1. which generator is best
2. whether dangerous-branch CABR is useful
3. whether calibration can make the system selective
4. whether the evolutionary population improves over time
5. whether the final system survives walk-forward validation

---

# What Has Already Been Achieved

## Implemented Infrastructure

The following V24.1 modules now exist:

```text
src/v24_1/validation_dataset.py
src/v24_1/branch_realism.py
src/v24_1/generator_tournament.py
src/v24_1/cabr_tradeability.py
src/v24_1/evolution_runner.py
src/v24_1/calibration_model.py
```

The system can now:

- generate many future branches
- score each branch
- identify dangerous invalidation branches
- compare multiple generator architectures
- evolve trading-agent populations
- calibrate whether a trade is truly worth taking

The project also already achieved:

- complete V24 architecture integration
- diffusion-based future generation
- Mamba/xLSTM experimental support
- OpenClaw integration path
- low-participation selective trading philosophy
- multi-horizon 15m / 30m strategic focus

---

# What Has Been Learned So Far

The research now strongly suggests:

```text
15m / 30m horizons > 5m horizon
```

The project appears to have more signal in slightly slower strategic horizons than in ultra-short-term prediction.

Also:

```text
selection + abstention + branch realism
    matter more than
raw direction accuracy
```

The project repeatedly showed that:

- ROC-AUC remains only slightly above chance
- but filtered selective trades can still become profitable
- therefore the real edge likely comes from:
  - selecting only the best states
  - avoiding dangerous minority branches
  - preserving uncertainty instead of hiding it

---

# Core Concept For The Next Phase

The new architecture should be:

```text
World State
    -> Multiple Future Generators
    -> Branch Realism Filter
    -> CABR Tradeability
    -> Calibration Model
    -> Evolutionary Strategy Selection
    -> Final Trade Decision
```

Where:

```text
tradeability =
    best_branch_score
    - dangerous_branch_score
```

and:

```text
true_trade_probability =
    probability that trade reaches +0.5R
    before reaching -1R
```

The system should only trade when both are high.

---

# V24.1 Immediate Next Steps

## Step 1 — Generator Tournament

Run:

```text
scripts/run_generator_tournament.py
```

Compare:

- Diffusion
- CVAE
- Transformer
- Mamba
- xLSTM

Metrics:

- cone containment
- minority rescue rate
- branch realism
- trade expectancy after CABR
- runtime

Expected output:

```text
outputs/v24_1/generator_leaderboard.json
```

Likely hypothesis:

```text
Mamba/xLSTM
    -> best central forecast

Diffusion
    -> best minority scenarios

Hybrid
    -> best overall result
```

---

## Step 2 — Dangerous-Branch CABR Validation

Run:

```text
scripts/test_tradeability.py
```

Goal:

Prove that:

```text
tradeability
```

predicts profitability better than raw confidence.

Important metric:

```text
correlation(tradeability, realized_R)
```

Target:

```text
tradeability correlation > confidence correlation
```

---

## Step 3 — Calibration Model

Train:

```text
src/v24_1/calibration_model.py
```

Target label:

```text
1 if trade reaches +0.5R before -1R
0 otherwise
```

Inputs:

- branch disagreement
- dangerous branch score
- analog agreement
- regime confidence
- spread/slippage estimate
- recent drawdown

Expected output:

```text
true_trade_probability
```

Then filter trades using:

```text
true_trade_probability > threshold
```

---

## Step 4 — Evolutionary Strategy Population

Only after calibration works.

Evolve:

- tradeability threshold
- calibration threshold
- regime preference
- stop-loss multiplier
- take-profit multiplier
- generator preference

The evolutionary system should optimize:

```text
expectancy
+ Sharpe
- drawdown
```

rather than raw win rate.

---

## Step 5 — Final Walk-Forward Validation

Run full validation on:

- 2023
- 2024
- 2025
- 2026

Success criteria:

```text
Participation: 2–10%
Win Rate: >60%
Expectancy: >0.25R
Max Drawdown: <20%
Cone Containment: >70%
```

If the system does not meet those targets:

```text
Do not build V25.
```

Instead:

- identify the failing subsystem
- retrain only that subsystem
- rerun validation

---

# Next Prompt

```md
# Nexus Trader V24.1 — Immediate Validation Directive

You are the lead research engineer for Nexus Trader V24.1.

The V24 architecture is already implemented.

Do NOT add more systems.

Your task is to determine which existing systems create a real trading edge.

Priority order:

1. Generator tournament
2. Dangerous-branch CABR validation
3. Calibration model
4. Evolutionary strategy tuning
5. Full walk-forward validation

The system must optimize:

- tradeability
- calibrated selectivity
- minority-branch preservation
- low participation with high expectancy

Never optimize only for:

- raw direction accuracy
- trade count
- confidence alone

Required rule:

```text
Only trade if:
    tradeability_score is high
AND true_trade_probability is high
```

The project should prefer:

```text
2–10% participation
65–75% win rate
>0.25R expectancy
```

over:

```text
50% participation
weak edge
```

Primary deliverables:

```text
outputs/v24_1/generator_leaderboard.json
outputs/v24_1/tradeability_report.json
outputs/v24_1/calibration_report.json
outputs/v24_1/evolution_history.json
outputs/v24_1/final_walkforward_report.json
```

If any subsystem fails:

- identify the failure precisely
- redesign only that subsystem
- do not continue to a new architecture version until validation succeeds
```

