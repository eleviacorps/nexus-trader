# Nexus Trader — V9 Codex Execution Directive

## Who You Are

You are the lead research engineer and implementation agent for **Nexus Trader**.

Nexus Trader is a market simulation and future-path prediction system for XAUUSD and related pairs. It is not a classic trading bot. It is a crowd-simulation engine that generates plausible future price paths, scores them for realism, and collapses them into a probability cone that a human trader watches in real time.

You have full authority to:

- redesign any module
- invent new scoring formulas
- create new source files
- refactor existing architecture
- abandon weak approaches
- propose and implement novel systems described in `CORE_CONCEPT_AND_THEORY.md`

You must never:

- fake backtest numbers
- optimise for cosmetic metric improvements
- optimise raw ROC-AUC as the primary goal
- treat LLM output as a numeric price predictor
- over-fit to a single recent regime

---

## Current Project State

Read `MASTER.md` in full before doing anything.

Key facts you must internalise from that file:

- raw TFT directional ROC-AUC is stuck near `0.51`
- the real edge comes from **selection + abstention + simulation structure**, not from classifier strength
- V8 proved that the branch selector is now the primary source of signal
- V8 full branch selector achieved:
  - top-1 branch accuracy: `61.3%`
  - top-3 containment: `63.6%`
  - event-driven 15m win rate: `52.4%`
- the `15m` horizon is consistently the strongest horizon
- `5m` raw directional accuracy is near-random and likely near its ceiling
- the cone is still too uniform in width — branch diversity is insufficient
- the news pipeline has never been fully built — model is price-feature-only
- the gate has a binary failure mode: either overtrades or takes zero trades

---

## V9 Primary Objective

```
Maximise realistic 15m future-path selection quality.
```

Do NOT optimise for:

- generic up/down prediction
- broad ROC-AUC
- optimistic backtests
- cosmetic win-rate improvements

Optimise only for:

```
better branch ranking
better top-3 containment
better event-driven 15m realism
better cone reliability
```

---

## V9 Target Metrics

| Metric | V8 Baseline | V9 Target |
|---|---|---|
| Top-1 branch accuracy | 61.3% | 68–75% |
| Top-3 containment | 63.6% | 78–82% |
| Event-driven 15m win rate | 52.4% | 55–60% |
| Cone containment rate | not yet measured | > 65% |
| Gate participation | ~18% | 20–30% non-dead |

---

## V9 Execution Phases

### PHASE 0 — Read And Verify State

Before writing a single line of code:

1. Read `MASTER.md` in full
2. Read `CORE_CONCEPT_AND_THEORY.md` in full
3. Verify the following artifacts exist:
   - `outputs/v8/branch_archive.parquet`
   - `models/tft/final_tft.ckpt`
   - `outputs/evaluation/v8_summary.json`
   - `data/features/fused_features.npy`
   - `config/project_config.py`
4. Print a brief status report of what is present and what is missing
5. Do not proceed until all critical artifacts are confirmed

---

### PHASE 1 — Reuse V8 Branch Archive

Do not retrain TFT yet.

Load:

```
outputs/v8/branch_archive.parquet
```

This archive contains per-timestamp records with:

- market state features
- 64 generated branches
- realised future path
- existing branch scores
- best branch label (V8 definition)

All V9 selector experiments must be built on this archive first.

Only regenerate TFT branches if selector improvements plateau after exhausting selector-only experiments.

---

### PHASE 2 — Improved Branch Labels

Create:

```
src/v9/branch_labels.py
```

Replace the V8 label (nearest raw path error only) with a composite score:

```
composite_score_i =
    + 0.35 * final_price_accuracy
    + 0.30 * full_path_similarity
    + 0.15 * execution_realism
    + 0.10 * regime_consistency
    + 0.10 * volatility_realism
```

Definitions:

- `final_price_accuracy` — inverse normalised error of predicted final 15m price vs realised
- `full_path_similarity` — inverse mean absolute error across all timesteps in the branch path
- `execution_realism` — whether the path is tradeable under realistic spread + slippage assumptions (default: 0.5 pip spread, 0.2 pip slippage for XAUUSD)
- `regime_consistency` — cosine similarity between branch implied regime vector and current HMM regime state
- `volatility_realism` — whether the branch move lies within ±2 sigma of the GARCH predicted envelope

Also produce these alternate label variants and store all of them:

- `top_1_branch` — index of branch with highest composite score
- `top_3_branches` — indices of top 3 by composite score
- `inside_confidence_cone` — boolean: did the realised path stay inside the branch envelope?
- `minority_rescue_branch` — the branch that would have been correct when the consensus was wrong

Save all labels to:

```
outputs/v9/branch_labels.parquet
```

---

### PHASE 3 — New Branch Features

Create:

```
src/v9/branch_features_v9.py
```

Compute the following per-branch features for every record in the archive:

**Path geometry:**

- `path_curvature` — mean second derivative of the path
- `path_acceleration` — mean rate of change of velocity
- `path_entropy` — Shannon entropy of the binned return distribution
- `path_smoothness` — inverse mean absolute change between consecutive steps
- `path_convexity` — whether the path is concave up or concave down
- `reversal_likelihood` — proportion of steps where direction reverses
- `breakout_likelihood` — proportion of steps exceeding 1 ATR from open
- `mean_reversion_likelihood` — proportion of steps returning toward open price

**Realism checks:**

- `news_consistency` — dot product of branch direction with current news sentiment vector
- `macro_consistency` — alignment of branch direction with current LLM macro thesis polarity
- `crowd_consistency` — alignment with current persona swarm aggregate direction
- `order_flow_plausibility` — whether branch volume profile matches historical volume shape at this session hour

**Analog features:**

- `analog_density` — count of historical analogs within DTW distance threshold
- `analog_disagreement` — standard deviation of analog outcome directions
- `analog_weighted_accuracy` — accuracy of analog predictions weighted by DTW similarity

**Regime features:**

- `hmm_regime_probability` — probability of current regime from HMM
- `regime_persistence` — how many consecutive bars the current regime has held
- `regime_transition_risk` — HMM transition probability away from current regime

**Quant features:**

- `garch_zscore` — branch final move as a z-score under current GARCH sigma
- `fair_value_distance` — distance of branch endpoint from Kalman fair value estimate
- `fair_value_mean_reversion_prob` — probability that the branch endpoint mean-reverts to fair value within 30m
- `atr_normalised_move` — branch total move divided by current ATR
- `historical_move_percentile` — percentile rank of branch move in historical move distribution

**Cross-branch features:**

- `branch_disagreement` — standard deviation of all 64 branch final prices
- `consensus_direction` — modal direction across all 64 branches
- `consensus_strength` — proportion of branches agreeing with modal direction

Then create interaction features:

```python
regime_match_x_analog = regime_consistency * analog_density
volatility_realism_x_fair_value = volatility_realism * (1 / (fair_value_distance + 1e-6))
news_x_crowd = news_consistency * crowd_consistency
analog_density_x_regime_persistence = analog_density * regime_persistence
```

Save branch features to:

```
outputs/v9/branch_features_v9.parquet
```

---

### PHASE 4 — Novel System Implementations

Read `CORE_CONCEPT_AND_THEORY.md` for full descriptions of each system.

Implement in priority order:

#### 4A — Live Persona Calibration Loop

Create:

```
src/v9/persona_calibration.py
```

After every 15m window closes, update persona capital weights using an exponential moving average:

```python
alpha = 0.05  # decay factor — recent performance matters more

for each persona p:
    was_correct = (persona_signal[p] == actual_direction)
    accuracy_ema[p] = alpha * was_correct + (1 - alpha) * accuracy_ema[p]

# Normalise weights so they sum to 1.0
capital_weight[p] = softmax(accuracy_ema)[p]
```

Store calibration history to:

```
outputs/v9/persona_calibration_history.parquet
```

Expose current weights via:

```
src/service/live_data.py  (add persona_weights field to simulation response)
```

#### 4B — Market Memory Bank

Create:

```
src/v9/memory_bank.py
```

Build a contrastive embedding space from the historical feature archive:

- encoder: 3-layer MLP, input = 60-bar feature window flattened, output = 64-dim embedding
- training objective: contrastive loss — windows with same 15m direction outcome are positive pairs
- training data: full historical fused features
- index: FAISS flat index over all embeddings

At inference:

```python
query_embedding = encoder(current_window)
top_k_indices, distances = index.search(query_embedding, k=20)
analog_outcome_votes = historical_directions[top_k_indices]
analog_confidence = weighted_mean(analog_outcome_votes, weights=1/distances)
```

Inject `analog_confidence` into:

- branch selector features
- persona calibration context
- live simulation response

Save trained encoder to:

```
checkpoints/memory_bank/encoder.pt
```

Save FAISS index to:

```
checkpoints/memory_bank/index.faiss
```

#### 4C — Cross-Timeframe Contradiction Detector

Create:

```
src/v9/contradiction_detector.py
```

Input: multi-horizon branch direction outputs (5m, 15m, 30m) and their confidence scores.

Classify each situation into:

```python
class ContradictionType(Enum):
    FULL_AGREEMENT_BULL    = "agreement_bull"
    FULL_AGREEMENT_BEAR    = "agreement_bear"
    SHORT_TERM_CONTRARY    = "short_term_contrary"   # 5m disagrees with 15m+30m → likely liquidity sweep
    LONG_TERM_CONTRARY     = "long_term_contrary"    # 5m+15m agree but 30m disagrees → possible reversal
    FULL_DISAGREEMENT      = "full_disagreement"     # no consensus at any horizon → abstain
```

Train a small XGBoost classifier on historical multi-horizon outputs with these class labels.

At inference, expose:

- contradiction type
- contradiction confidence
- recommended cone treatment:
  - `full_agreement` → normal cone
  - `short_term_contrary` → flag as possible liquidity sweep in UI
  - `long_term_contrary` → flag as possible fake breakout
  - `full_disagreement` → widen cone, reduce signal confidence, recommend abstain

Add contradiction type as a feature to the branch selector.

#### 4D — Regret-Minimizing Abstention Gate

Create:

```
src/v9/regret_gate.py
```

Replace the binary gate with an asymmetric cost function:

```python
def regret_gate(pred_direction, pred_confidence, projected_move_pips, spread_pips=0.5):
    # Cost of being wrong: lose the projected move + spread
    cost_of_wrong = projected_move_pips + spread_pips

    # Cost of missing: lose the projected move (opportunity cost)
    cost_of_missing = projected_move_pips * pred_confidence

    # Trade only if expected regret of wrong < expected regret of missing
    expected_regret_trade  = cost_of_wrong * (1 - pred_confidence)
    expected_regret_abstain = cost_of_missing

    should_trade = expected_regret_trade < expected_regret_abstain
    regret_margin = expected_regret_abstain - expected_regret_trade

    return should_trade, regret_margin
```

The `regret_margin` becomes a continuous signal that can replace the hard threshold. Higher margin = more confident the trade is worth taking.

Replace the existing gate in:

```
src/training/meta_gate.py
src/evaluation/walkforward.py
```

---

### PHASE 5 — Selector Experiments

Create:

```
src/v9/selector_experiments.py
```

Run all variants using the V9 branch labels and features from Phases 2 and 3.

#### Selector A — Improved XGBoost

```python
XGBClassifier(
    objective='multi:softprob',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss',
    early_stopping_rounds=30,
)
```

#### Selector B — Pairwise Ranking

```python
# Use LightGBM lambdarank objective
# For each timestamp, create pairwise comparisons of all 64 branches
# Label: 1 if branch_i is better than branch_j by composite score
LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    ndcg_eval_at=[1, 3],
    n_estimators=300,
    num_leaves=63,
)
```

#### Selector C — Regime-Specific Selectors

Train one XGBoost selector per HMM regime class. At inference, route to the matching regime selector. If regime confidence is below 0.6, use the ensemble selector instead.

#### Selector D — Top-3 Optimizer

Objective: maximise probability that the true best branch is inside the top-3 ranked branches. Use a custom loss that penalises true-branch rank > 3 much more heavily than rank 2 vs rank 1.

#### Selector E — Analog-Heavy Selector

Feature set restricted to: `analog_density`, `analog_disagreement`, `analog_weighted_accuracy`, `memory_bank_confidence`, `dtw_distance_rank`. Measures whether analog information alone is sufficient for selection.

#### Selector F — Quant Rejection Selector

Feature set restricted to: `garch_zscore`, `fair_value_distance`, `regime_consistency`, `volatility_realism`, `hmm_regime_probability`. Acts as a reality filter — rejects physically implausible branches before ranking begins.

#### Selector G — Full Ensemble

Combine predictions from Selectors A through F:

```python
final_score = (
    0.25 * selector_a_prob +
    0.20 * selector_b_score +
    0.20 * selector_c_prob +
    0.15 * selector_d_prob +
    0.10 * selector_e_prob +
    0.10 * selector_f_prob
)
# Weights learned via simple logistic regression on validation fold
```

---

### PHASE 6 — Evaluation

For every selector variant produce:

```
outputs/v9/selector_experiment_results.json
outputs/v9/selector_experiment_results.md
```

Report per variant:

- top-1 branch accuracy
- top-3 containment rate
- confidence cone containment rate
- event-driven 15m win rate
- event-driven 15m avg unit PnL
- regime-by-regime accuracy breakdown
- minority branch rescue rate
- trade count at target participation level

Selector priority order when choosing winner:

```
1. top-3 containment
2. event-driven 15m quality
3. top-1 accuracy
4. regime robustness across all regime classes
```

Do not choose a selector based on top-1 alone.

---

### PHASE 7 — Quant Stack Upgrades

Only run this phase after baseline selector experiments complete. Upgrade quant models, rerun best selector, measure delta.

#### HMM Upgrade

File: `src/v8/hmm_regime.py`

Add:

- semi-Markov persistence modelling
- transition probability matrix
- regime confidence output
- uncertain regime state handling
- output: `{top_regime, second_regime, regime_entropy, transition_risk}`

#### Volatility Upgrade

File: `src/v8/garch_volatility.py`

Compare GARCH vs EGARCH vs realised-volatility baseline. Keep strongest. Add:

- branch volatility percentile
- branch sigma distance from predicted envelope

#### Analog Retrieval Upgrade

File: `src/v8/analog_retrieval.py`

Replace single-row similarity with:

- multi-bar DTW distance
- nearest-neighbour in Memory Bank embedding space
- weighted analog voting across top-5, top-20, and full neighbourhood
- store which retrieval method has highest historical accuracy per regime

---

### PHASE 8 — Conditional Full Retrain

Only launch if selector improvements plateau and branch archive quality is the limiting factor.

If retraining:

- keep TFT frozen initially
- fine-tune only on 15m branch generation quality
- regenerate branch archive using upgraded quant stack
- retrain selector on new archive

Do not spend another 26-hour compute cycle unless the selector clearly hits its ceiling on the existing archive.

The retrain should target:

- better branch diversity (wider cone by construction, not by post-hoc noise injection)
- more regime-specific branches
- better minority branch generation
- news and macro conditioned branch variation

---

### PHASE 9 — Live Integration

After strongest selector is confirmed:

Update:

```
src/service/live_data.py
src/ui/*
```

The UI must show:

- top branch with explanation
- top 3 branches overlaid
- minority risk branch
- confidence cone with inner epistemic layer and outer aleatoric layer
- contradiction type label when applicable
- persona weight distribution (live calibration state)
- which quant models supported or rejected the chosen branch

Branch explanation format:

```json
{
  "chosen_branch": 14,
  "reason": {
    "hmm_regime": "strong trend match",
    "garch": "move within 1.2 sigma envelope",
    "analog": "18 of 20 analogs support direction",
    "news": "consistent with current sentiment",
    "contradiction": "none — full agreement across 5m/15m/30m"
  },
  "minority_branch": 31,
  "minority_reason": "only branch consistent with 30m structure break scenario"
}
```

---

### PHASE 10 — Final Reports

Create:

```
outputs/evaluation/v9_summary.json
outputs/evaluation/v9_summary.md
```

The report must include:

- V8 vs V9 metric comparison (all metrics from Phase 6)
- which selector variant won and why
- which features had highest importance in the winning selector
- which regimes remain hardest to predict
- whether TFT retraining was necessary
- whether quant stack upgrades improved selector meaningfully
- whether novel systems (persona calibration, memory bank, contradiction detector, regret gate) each contributed measurable lift
- honest interpretation: what improved, what did not, what remains unsolved

Update `MASTER.md`:

- append V9 architecture summary
- append V9 final metrics
- append honest V9 interpretation
- append V10 recommendation

---

## Explicit Authority

You are explicitly authorised to:

- redesign selector objectives and loss functions
- create new branch label formulas
- invent new quant scoring methods
- add new branch features not listed above
- abandon weak approaches mid-phase
- ignore 5m if 15m is clearly superior
- use ranking loss instead of classification
- build ensembles of any depth
- keep TFT frozen if retraining adds no value
- implement any novel system described in `CORE_CONCEPT_AND_THEORY.md`

You are not authorised to:

- fake backtest results
- report metrics on training data as validation performance
- treat ROC-AUC alone as success
- claim 90%+ accuracy without walk-forward proof
- add dependencies without documenting them in `requirements.txt`

---

## File Naming Convention

```
src/v9/              — all new V9 source modules
outputs/v9/          — all V9 intermediate outputs
outputs/evaluation/  — all V9 final reports
checkpoints/v9/      — all V9 model checkpoints
```

Never overwrite V8 artifacts. Always write V9 outputs to new paths.

---

## Reminders

- `15m` is the primary product horizon — optimise for it explicitly
- the cone width being too uniform is a known bug — diversity improvement is a first-class goal
- the gate binary failure mode (zero trades vs overtrades) must be resolved — the regret gate is the intended fix
- news pipeline is still missing — add news consistency as a branch feature before selector training
- persona calibration is nearly free to implement — do it early
- always measure before and after — every phase must produce a delta report
