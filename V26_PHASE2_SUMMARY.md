# V26 Phase 2: Multi-Horizon Path Stacking — Implementation Summary

## Overview
Phase 2 extends the Phase 1 regime-conditioned generator with **hierarchical multi-horizon path stacking**, enabling coherent futures across short/medium/long timeframes.

## Files Created

### Core Components
| File | Purpose | Lines |
|------|---------|-------|
| `src/v26/diffusion/multi_horizon_generator.py` | Hierarchical multi-horizon generator | 414 |
| `src/v26/diffusion/horizon_stack.py` | Path stacking and boundary blending | 278 |
| `src/v26/diffusion/regime_embedding.py` | Regime embedding (from Phase 1) | 135 |
| `src/v26/diffusion/regime_generator.py` | Regime-conditioned generator (from Phase 1) | 433 |
| `src/v26/__init__.py` | V26 module exports | 12 |
| `src/v26/diffusion/__init__.py` | Diffusion submodule exports | 20 |

### Training & Evaluation
| File | Purpose | Lines |
|------|---------|-------|
| `scripts/train_v26_multi_horizon.py` | Multi-horizon training script | 267 |
| `scripts/evaluate_v26_phase2.py` | Phase 2 evaluation | 347 |

## Architecture

### Multi-Horizon Hierarchy
```
past context → short-horizon (14 paths, 120 bars)
                   ↓
            short_summary (64-dim)
                   ↓
         medium-horizon (4 paths, 120 bars)
                   ↓
            medium_summary (64-dim)
                   ↓
          long-horizon (2 paths, 120 bars)
```

### Key Features
1. **Horizon Summary Encoders**: Compress generated paths into latent summaries
   - `HorizonSummaryEncoder`: (B, C, L) → (B, 64) via conv + pooling
   
2. **Latent Conditioning**: Higher horizons receive lower-horizon summaries
   - Medium: temporal_emb + regime_emb + short_summary
   - Long: temporal_emb + regime_emb + short_summary + medium_summary

3. **Path Distribution**: 70/20/10 split across top-3 regimes per horizon
   - Preserves minority futures (unlike single-regime generation)

4. **Boundary Blending**: Smooth transitions between horizons
   - Linear blend of overlapping regions (default 10 bars)

## Training

### From Phase 1 Checkpoint
```bash
python scripts/train_v26_multi_horizon.py \
    --epochs 5 \
    --batch-size 32 \
    --lr 5e-6 \
    --phase1-ckpt models/v26/diffusion_phase1_final.pt \
    --output models/v26/diffusion_phase2_multi_horizon.pt
```

### Training Loss Components
1. **Horizon Consistency Loss**: Maximize consistency between horizons
2. **Regime Preservation Loss**: KL divergence from target regime distribution
3. **Combined**: `loss = consistency_loss + 0.1 * regime_loss`

## Evaluation

### Run Evaluation
```bash
python scripts/evaluate_v26_phase2.py \
    --phase1-ckpt models/v26/diffusion_phase1_final.pt \
    --phase2-ckpt models/v26/diffusion_phase2_multi_horizon.pt \
    --output outputs/v26/phase2_evaluation_report.json
```

### Metrics
| Metric | Phase 1 Baseline | Phase 2 Target | Description |
|--------|------------------|----------------|-------------|
| realism_score | 0.5538 | >0.60 | Overall path realism |
| regime_consistency | 100% | >90% | Regime conditioning fidelity |
| distinct_separation | 0.4433 | >0.45 | Distinct regime characteristics |
| horizon_consistency | N/A | >0.75 | Cross-horizon coherence |

### Success Criteria
```
realism_score > 0.60
regime_consistency > 90%
distinct_separation > 0.45
horizon_consistency_score > 0.75
```

## Integration Path (SANDBOXED)

Phase 2 integrates with V25 via `src/v26/integration/branch_router.py`:

```
V26 Multi-Horizon Paths
        ↓
HorizonStack.stack_paths()
        ↓
V26BranchRouter.route_branches()
        ↓
V25 CABR + RegimeSpecificRanker
        ↓
ExecutionDecision (sandboxed)
```

**SAFETY**: V25 integration is DISABLED until Phase 2 evaluation shows clear improvement over Phase 1 baseline.

## Next Steps

1. **Run training**: Fine-tune for 5 epochs from Phase 1 checkpoint
2. **Evaluate**: Compare Phase 1 vs Phase 2 metrics
3. **Decision**:
   - If targets met → Promote to V25 integration
   - If not met → Additional training or architecture adjustments

## Phase 1 Baseline (Preserved)
- Checkpoint: `models/v26/diffusion_phase1_final.pt`
- Epoch: 7, Val Loss: 1.109108
- Realism Score: 0.5538
- Regime Consistency: 100%

## Notes
- Phase 1 checkpoint is FROZEN and will not be overwritten
- Phase 2 training only optimizes new components (summary encoders, projection layers)
- Base generator remains frozen to preserve Phase 1 capabilities