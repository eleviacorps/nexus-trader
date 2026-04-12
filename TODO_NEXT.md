# V24 Phases 2-6 Implementation Plan

## Context

Nexus Trader V24 represents a fundamental architectural shift from directional prediction to **trade quality prediction**. Phase 1 (the V24 Bridge) successfully established a heuristic implementation that achieves 80% win rate on 2023-12 and 62% on 2024-12 by framing execution as a quality-based decision rather than direction alone.

This plan covers Phases 2-6: the complete implementation of learned models (meta-aggregator, diffusion generator, dual CABR) and multi-agent architecture (evolutionary population, OpenClaw supervisor).

**Current State:** Phase 1 Bridge ✓ Complete  
**Target State:** Full V24 Architecture with learned components

---

## Phase 2: Learned Meta-Aggregator

### Objective
Replace the heuristic quality scoring in `src/v24/meta_aggregator.py` with a learned model that predicts:
- Expected trade value (primary target)
- Win probability
- Realized R:R
- Danger score
- Uncertainty
- Abstain probability

### Files to Create/Modify

#### New Files
- `src/v24/models/meta_aggregator_model.py` (~400 lines)
  - Transformer/xLSTM mixture-of-experts architecture
  - Regime-aware gating network
  - Multi-head output for quality metrics
  
- `scripts/train_v24_meta_aggregator.py` (~350 lines)
  - Training loop with trade-quality targets
  - Validation on held-out months
  - Checkpoint saving with regime-conditioned metrics

- `src/v24/training/target_builder.py` (~200 lines)
  - Build expected_value = P(win)*reward - P(loss)*loss
  - Compute danger scores from historical loss distributions
  - Normalize uncertainty from prediction variance

#### Modifications
- `src/v24/meta_aggregator.py` (~200 lines change)
  - Add loading of learned weights
  - Fallback to heuristic if checkpoint missing

- `config/project_config.py`
  - Add paths: `V24_META_AGGREGATOR_PATH`, `V24_META_AGGREGATOR_CONFIG`

### Training Data
- Input: Month debug outputs (`outputs/v22/month_debug_suite_*.json`)
- Augmented SJD dataset: `outputs/v22/sjd_dataset_v22_augmented.parquet` (215 rows)
- Aligned features: `data/features/fused_features.npy` + `gate_context.npy`

### Training Targets
```python
{
    "expected_value": float,       # Primary target
    "win_probability": float,      # Auxiliary
    "realized_rr": float,          # Auxiliary  
    "danger_score": float,         # Auxiliary
    "uncertainty": float,          # Auxiliary
    "abstain_probability": float   # Auxiliary
}
```

### Success Criteria
- Expected value predictions correlate >0.5 with realized outcomes
- Model beats heuristic bridge on 2024-12 (62% win rate baseline)
- No trade frequency collapse (<50 trades/month)

---

## Phase 3: Conditional Diffusion Future Generator

### Objective
Replace the synthetic branch generator with a learned diffusion model that generates 32-64 plausible future price paths conditioned on current state.

### Files to Create

#### Core Diffusion Models
- `src/v24/diffusion/unet_1d.py` (~300 lines)
  - 1D U-Net with time conditioning
  - Attention blocks for cross-modal conditioning
  
- `src/v24/diffusion/scheduler.py` (~150 lines)
  - DDPM or DDIM noise schedule
  - Sampling strategies for path generation

- `src/v24/diffusion/conditional_generator.py` (~400 lines)
  - Main diffusion model class
  - Condition on: world_state, regime_embedding, analog_memory
  - Generate OHLCV paths with volatility envelopes

- `src/v24/diffusion/world_model.py` (~350 lines)
  - Mamba-based world model alternative
  - Compare against diffusion in backtests

#### Training
- `scripts/train_v24_diffusion.py` (~400 lines)
  - Conditional diffusion training loop
  - Path realism metrics (volatility clustering, tail behavior)
  - Comparison training: CVAE, Transformer baselines

### Generated Outputs
```python
{
    "paths": [  # 32-64 paths
        {
            "ohlc": np.array,           # [timesteps, 4]
            "volatility_envelope": np.array,
            "drawdown_path": np.array,
            "branch_confidence": float
        }
    ]
}
```

### Success Criteria
- Generated paths show volatility clustering
- Tail events (3+ sigma) occur with realistic frequency
- Backtest expectancy using generated paths >= heuristic paths

---

## Phase 4: CABR V24 — Dual Branch Ranking

### Objective
Implement dual scoring: `best_branch_score` (most profitable) and `dangerous_branch_score` (most risky), with abstain when danger > best.

### Files to Create/Modify

#### Core Dual Scorer
- `src/v24/cabr/dual_ranking.py` (~350 lines)
  - Separate networks for best vs dangerous branch
  - Danger score computed from component risk factors
  - Abstain logic: `if danger_score > best_score * safety_factor`

- `src/v24/cabr/risk_decomposition.py` (~200 lines)
  - Decompose danger into:
    - Regime transition risk
    - Tail risk (extreme moves)
    - Volatility regime mismatch
    - Analog disagreement
    - Model confidence collapse

#### Training & Evaluation
- `scripts/train_v24_cabr_dual.py` (~300 lines)
  - Train dual scorer on historical trades
  - Classify branches as: profitable, dangerous, neutral

- `src/v24/cabr/safety_calculator.py` (~150 lines)
  - Compute stop placement from dangerous branch
  - Position sizing from danger/best ratio

### Merged Outputs
```python
{
    "best_branch": {
        "score": float,
        "path_index": int,
        "expected_value": float
    },
    "dangerous_branch": {
        "score": float, 
        "path_index": int,
        "danger_factors": dict
    },
    "abstain_recommended": bool,
    "safety_margin": float  # best - danger
}
```

### Success Criteria
- Dangerous branch is identified in >80% of eventual loss trades
- Abstain rate increases during high-volatility regimes
- Win rate of executed trades improves vs baseline

---

## Phase 5: Evolutionary Agent Population

### Objective
Replace single strategy with population of 5-10 competing agents that evolve weekly based on fitness.

### Files to Create

#### Agent Framework
- `src/v24/agents/base_agent.py` (~200 lines)
  - Abstract base class for all agents
  - Policy method: `decide(world_state) -> Action`

- `src/v24/agents/trend_follower.py` (~150 lines)
- `src/v24/agents/mean_reversion.py` (~150 lines)
- `src/v24/agents/macro_heavy.py` (~150 lines)
- `src/v24/agents/analog_heavy.py` (~150 lines)
- `src/v24/agents/high_abstention.py` (~150 lines)
- `src/v24/agents/aggressive.py` (~150 lines)
- `src/v24/agents/low_risk.py` (~150 lines)
- `src/v24/agents/breakout.py` (~150 lines)

Each implements different threshold/regime/weighting logic.

#### Evolution System
- `src/v24/evolution/population.py` (~300 lines)
  - Manage agent population
  - Track performance statistics
  
- `src/v24/evolution/scheduler.py` (~200 lines)
  - Weekly evolution cycle
  - Selection: bottom 30% removed
  - Reproduction: top 20% copied
  - Mutation: parameter perturbation

- `src/v24/evolution/fitness.py` (~150 lines)
  ```python
  fitness = 0.35 * expectancy + 0.20 * sharpe - 0.20 * drawdown + 0.15 * stability + 0.10 * frequency
  ```

- `src/v24/evolution/mutation.py` (~200 lines)
  - Confidence threshold ±delta
  - Regime sensitivity scaling
  - TP/SL multiplier adjustment
  - Danger weighting shift
  - Macro weighting shift

#### Runner
- `scripts/run_v24_evolution.py` (~250 lines)
  - Weekly evolution cycle runner
  - Performance tracking and logging

### State Store
`outputs/v24/evolution/population_state.json`

### Success Criteria
- Fitness improves over 4+ weeks of evolution
- Population diversity maintained (>5 distinct phenotypes)
- Best agent beats single-strategy baseline

---

## Phase 6: OpenClaw Supervisor

### Objective
Add supervisory layer that monitors macro, geopolitical, and system health to select active agent families and suggest mutations.

### Files to Create

#### Core Supervisor
- `src/v24/openclaw/supervisor.py` (~400 lines)
  - Monitor feeds: news, macro, geopolitical
  - Select agent families based on regime
  - Suggest mutations to evolution scheduler
  - Pause trading during extreme events

- `src/v24/openclaw/news_monitor.py` (~250 lines)
  - GDELT event monitoring
  - Federal Reserve/FOMC watcher
  - Geopolitical stress scoring

- `src/v24/openclaw/macroeconomic_state.py` (~200 lines)
  - FRED API integration (rates, spreads, conditions)
  - Real-time macro regime classification

- `src/v24/openclaw/approval_layer.py` (~200 lines)
  - Three modes: research, assisted, autonomous
  - Human approval interface
  - Override safety controls

#### Configuration
- `config/openclaw_config.yaml` (~100 lines)
  - Event severity thresholds
  - Pause trading triggers
  - Agent family selection rules

### Outputs
```python
{
    "macro_regime": str,  # risk_on, risk_off, shock, calm
    "geopolitical_stress": float,  # 0-1
    "system_health": {
        "recent_drawdown": float,
        "circuit_breaker_state": str,
        "agent_performance": dict
    },
    "suggested_agent_family": str,
    "suggested_mutations": list,
    "trading_paused": bool,
    "pause_reason": str
}
```

### Success Criteria
- Detected stress events correlate with market volatility
- Pause triggers prevent >5% drawdown during high-stress
- Agent family selection improves aggregate fitness

---

## Implementation Order & Dependencies

```
Phase 2: Learned Meta-Aggregator
    │
    ▼ (uses world_state from Phase 1)
Phase 3: Conditional Diffusion Generator  ──┐
    │                                        │
    ▼                                        │
Phase 4: CABR V24 Dual Scoring              │
    │  (uses diffusion paths)               │
    │  (uses meta-aggregator outputs)      │
    ▼                                        │
Phase 5: Evolutionary Agent Population  ◄─────┘
    │  (uses dual CABR scoring)
    │  (uses meta-aggregator for fitness)
    ▼
Phase 6: OpenClaw Supervisor
       (monitors all lower layers)
       (controls agent selection)
```

---

## Testing Strategy

### Unit Tests
- Each module needs tests in `tests/test_v24_*.py`
- Minimum coverage: 70% for learned components

### Integration Tests
- `tests/integration/test_v24_pipeline.py`
- Full pipeline: WorldState → MetaAgg → Diffusion → CABR → Agents

### Validation Backtests
- Held-out months: 2024-12, 2025-Q1
- Track: Win rate, expectancy, Sharpe, max drawdown
- Compare against Phase 1 bridge baseline

---

## Checkpoint Artifacts

| Component | Path | Size Estimate |
|-----------|------|--------------|
| Meta-Aggregator | `models/v24/meta_agg_*.pt` | ~50-100MB |
| Diffusion Model | `models/v24/diffusion_*.pt` | ~200-500MB |
| CABR Dual | `models/v24/cabr_dual_*.pt` | ~20-50MB |
| Evolution State | `outputs/v24/evolution/*.json` | ~10MB |
| OpenClaw Config | `config/openclaw_*.yaml` | ~10KB |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Learned model overfits | Heavy dropout, early stopping, regime-based validation |
| Diffusion generates unrealistic paths | Add physical constraints, volatility clustering loss |
| Evolution converges to one phenotype | Maintain diversity bonus, mutation rate floor |
| OpenClaw false positives | Conservative thresholds, human override |
| Training takes too long | Use cloud MI300X, checkpoint frequently |

---

## Success Criteria for All 5 Phases

- **Phase 2:** Learned meta-aggregator beats heuristic by >5pp win rate
- **Phase 3:** Generated paths show realistic volatility clustering
- **Phase 4:** Dangerous branch identified in >80% of losses
- **Phase 5:** Best evolved agent beats single-strategy by >10% expectancy
- **Phase 6:** System pauses during stress events, reducing drawdown

**Final V24 Goal:** 
- Win rate >70% on held-out months
- Trade frequency: 50-150/month
- Sharpe-like >2.0
- Max drawdown <15%
