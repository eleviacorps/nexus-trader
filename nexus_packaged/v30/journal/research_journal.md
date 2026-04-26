# V30 Research Journal

## Overview

V30 introduces a learnable **Evaluator** that scores diffusion-generated paths based on their similarity to actual market behavior. Instead of picking a single path or using heuristic signals, V30 learns a weighted distribution over paths.

## Architecture

```
GENERATOR (frozen) → 64 future paths
                        ↓
EVALUATOR (trainable) → weights per path [0-1]
                        ↓
AGGREGATOR → prob_up, EV, uncertainty, confidence → decision
```

## Key Innovations

1. **Distribution-level learning**: No argmax - learn weighted belief
2. **Differentiable scoring**: Structure, magnitude, direction similarity
3. **Softmax weighting**: Stable gradients, no path identity overfitting
4. **Modular design**: Easy to swap components

## Experiments

### Entry Template

```
## [DATE] Experiment #

Config:
- num_paths:
- horizon:  
- loss_weights:

Results:
- avg_reward:
- directional_accuracy:
- calibration_error:

Observations:
- what improved
- what failed
- anomalies

Next Actions:
- changes to try
```

---

## Experiment 1: Baseline

Date: 2026-04-24

Config:
- num_paths: 64
- horizon: 20
- loss_weights: direction=0.4, magnitude=0.3, structure=0.3

Results:
- avg_reward: TBD
- directional_accuracy: TBD  
- calibration_error: TBD

Observations:
- Initial implementation complete
- Training pipeline ready
- Need to run first training experiment

Next Actions:
- Run training with small subset
- Validate path generation works
- Check reward computation

---

## Fixes Applied (2026-04-24)

### 1. CRITICAL: feature_dim bug fixed
```python
# BEFORE (wrong):
feature_dim=eval_cfg.get("hidden_dim", 256)

# AFTER (correct):
feature_dim=model_cfg.get("feature_dim", 144)
```

### 2. Confidence improved (FIXED)
```python
# BEFORE (weak):
confidence = abs(prob_up - 0.5) * 2

# AFTER (edge/risk ratio):
confidence = abs(expected_return) / uncertainty
```

### 3. Decision logic improved (FIXED)
```python
# BEFORE (dual condition):
if confidence >= threshold and abs(ev) >= threshold:
    BUY/SELL

# AFTER (score-based):
score = expected_return / uncertainty
if score > threshold: BUY
elif score < -threshold: SELL
else: HOLD
```

### 4. Generator efficiency (FIXED)
- Created `efficient_generator.py` with caching
- Added `CachedPathGenerator` for pre-computing paths
- Added `DeterministicPathGenerator` for reproducibility

### 5. Debug metrics already present
- entropy of weights ✓
- max weight ✓
- correlation(weights, rewards) ✓
- direction_accuracy ✓

### 6. Added reward distribution tracking
Added to training loop for debugging:
- rewards.mean()
- rewards.std()

---

## Notes

- Free APIs exhausted: TwelveData hit 800/day limit
- System running with OHLC fallback (4493.448)
- V30 provides path forward: learn which simulated futures resemble reality