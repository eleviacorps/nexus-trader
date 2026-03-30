# TODO_NEXT

## Notebook 08 - Confluence Gate

- Add the post-forecast confluence gate that combines fused TFT probability, persona disagreement, dominant driver, and macro context into a final actionability score.
- Define explicit suppression rules for wide-cone / low-consensus states.
- Persist the confluence decision and rationale so the UI can expose why a forecast was gated down.

## Notebook 09 - Backtesting

- Build a backtest harness around the fused 1-minute pipeline instead of the legacy price-only flow.
- Evaluate at least three policies: always-follow, consensus-thresholded, and cone-width-filtered.
- Track precision, recall, expectancy, max drawdown, and behavior across low/high-volatility regimes.
- Add a scenario report for macro shock windows and crowd-extreme windows.

## Notebook 10 - Chart UI

- Convert the architecture SVG concepts into the live probability cone chart.
- Surface current price, bullish probability, cone width, consensus score, dominant driver, and persona breakdown together.
- Keep the UI explanatory rather than signal-only: disagreement should remain visible.
- Add server-friendly loading states for missing embeddings, missing checkpoint, and stale feature matrices.
