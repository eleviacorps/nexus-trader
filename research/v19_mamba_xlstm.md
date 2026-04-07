# V19 Mamba/xLSTM Research Track

## Scope

This document is the research-only plan for the V19 long-range backbone work.
It does not replace the live V18/V19 production simulator path.

## Objective

Train a modern sequence model on the long XAUUSD minute archive with:

- 120-240 recent bars
- OHLCV and volatility state
- MMM features
- WLTC-derived crowd state
- MFG disagreement proxies

The target outputs are:

- 15-minute direction probability
- 30-minute direction probability
- volatility envelope
- regime class

## Candidate Backbones

1. Mamba-2
2. xLSTM
3. Hybrid Mamba encoder + CABR selector

## Why This Is Research-Only

The live simulator already has a functioning production path:

TFT generator -> CABR selector -> reverse collapse -> V19 local SJD judge

The Mamba/xLSTM line is a replacement candidate only after walk-forward evidence
beats the current branch-based stack on both participation and risk-adjusted quality.

## Evaluation Plan

1. Train on the long archive with chronological splits.
2. Compare against the current generator on:
   - 15-minute directional Brier score
   - regime classification accuracy
   - cone realism and volatility calibration
3. If the backbone improves the generator, keep CABR as the branch selector and
   evaluate the combined stack in walk-forward before any production routing change.

## Implementation Hook

The code scaffold lives in `src/v19/mamba_backbone.py`.
