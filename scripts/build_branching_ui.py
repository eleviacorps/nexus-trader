from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f'pandas is required for UI generation: {exc}')

from config.project_config import (  # noqa: E402
    FINAL_DASHBOARD_HTML_PATH,
    FUTURE_BRANCHES_PATH,
    LATEST_MARKET_SNAPSHOT_PATH,
    PERSONA_OUTPUTS_PATH,
    PERSONA_WEIGHT_HISTORY_PATH,
    PERSONA_BREAKDOWN_HTML_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    PROBABILITY_CONE_HTML_PATH,
)
from src.mcts.cone import build_probability_cone  # noqa: E402
from src.mcts.reverse_collapse import reverse_collapse  # noqa: E402
from src.mcts.tree import dominant_persona_name, expand_binary_tree, iter_leaves  # noqa: E402
from src.pipeline.fusion import load_price_frame  # noqa: E402
from src.simulation.personas import default_personas  # noqa: E402
from src.ui.render import render_persona_breakdown, render_probability_dashboard, write_branches_json  # noqa: E402


def resolve_price_frame():
    for path in [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV]:
        if path.exists():
            frame = load_price_frame(path)
            frame.index = pd.to_datetime(frame.index, errors='coerce')
            frame = frame[~frame.index.isna()].sort_index()
            return frame, path
    raise FileNotFoundError('No canonical price feature artifact available for UI generation.')


def apply_latest_persona_weights(personas):
    if not PERSONA_WEIGHT_HISTORY_PATH.exists():
        return personas, {}
    weights = pd.read_parquet(PERSONA_WEIGHT_HISTORY_PATH).tail(1)
    if weights.empty:
        return personas, {}
    snapshot = {}
    row = weights.iloc[0]
    for name, persona in personas.items():
        key = f'{name}_weight'
        if key in row.index:
            weight = float(row[key])
            persona.capital_weight = max(0.01, weight)
            snapshot[name] = weight
    return personas, snapshot


def latest_context(price_frame):
    row = price_frame.iloc[-1].to_dict()
    row['close'] = float(price_frame.iloc[-1].get('close', 0.0))
    if PERSONA_OUTPUTS_PATH.exists():
        persona_df = pd.read_parquet(PERSONA_OUTPUTS_PATH).tail(1)
        if not persona_df.empty:
            for key, value in persona_df.iloc[0].items():
                if key == 'timestamp':
                    continue
                if isinstance(value, (int, float)):
                    row[key] = float(value)
                else:
                    row[key] = value
    return row


def main() -> int:
    price_frame, price_path = resolve_price_frame()
    personas = default_personas()
    personas, weight_snapshot = apply_latest_persona_weights(personas)
    current_row = latest_context(price_frame)

    root = expand_binary_tree(current_row, personas, max_depth=5)
    leaves = iter_leaves(root)
    collapse = reverse_collapse(leaves)
    cone = build_probability_cone(collapse, horizon_steps=max(5, len(leaves[0].path_prices) if leaves and leaves[0].path_prices else 5))

    last_timestamp = pd.Timestamp(price_frame.index[-1])
    last_price = float(price_frame.iloc[-1].get('close', 0.0))
    branches = []
    for index, leaf in enumerate(sorted(leaves, key=lambda item: item.probability_weight, reverse=True), start=1):
        path_prices = leaf.path_prices or ([leaf.state.close] if leaf.state is not None else [last_price])
        timestamps = [(last_timestamp + timedelta(minutes=step)).isoformat() for step in range(1, len(path_prices) + 1)]
        branches.append(
            {
                'path_id': index,
                'probability': round(float(leaf.probability_weight), 6),
                'predicted_prices': [round(float(price), 5) for price in path_prices],
                'timestamps': timestamps,
                'dominant_persona': dominant_persona_name(leaf.state) if leaf.state is not None else 'unknown',
                'dominant_driver': leaf.dominant_driver,
            }
        )

    cone_points = []
    for point in cone.points:
        center_price = last_price * (1.0 + ((point.center - 0.5) * 0.02 * point.horizon))
        lower_price = max(0.0, last_price * (1.0 + ((point.lower - 0.5) * 0.02 * point.horizon)))
        upper_price = max(center_price, last_price * (1.0 + ((point.upper - 0.5) * 0.02 * point.horizon)))
        cone_points.append(
            {
                'horizon': point.horizon,
                'timestamp': (last_timestamp + timedelta(minutes=point.horizon)).isoformat(),
                'center_probability': point.center,
                'lower_probability': point.lower,
                'upper_probability': point.upper,
                'center_price': round(center_price, 5),
                'lower_price': round(lower_price, 5),
                'upper_price': round(upper_price, 5),
            }
        )

    if not weight_snapshot and PERSONA_OUTPUTS_PATH.exists():
        persona_df = pd.read_parquet(PERSONA_OUTPUTS_PATH).tail(1)
        if not persona_df.empty:
            row = persona_df.iloc[0]
            for name in ['retail', 'institutional', 'algo', 'whale', 'noise']:
                impact_key = f'{name}_impact'
                if impact_key in row.index:
                    weight_snapshot[name] = float(row[impact_key])

    write_branches_json(branches, FUTURE_BRANCHES_PATH)
    render_persona_breakdown(weight_snapshot, PERSONA_BREAKDOWN_HTML_PATH)
    render_probability_dashboard(price_frame, branches, cone_points, collapse.consensus_score, collapse.dominant_driver, weight_snapshot, FINAL_DASHBOARD_HTML_PATH)

    snapshot = {
        'price_source': str(price_path),
        'current_timestamp': last_timestamp.isoformat(),
        'current_price': last_price,
        'mean_probability': collapse.mean_probability,
        'uncertainty_width': collapse.uncertainty_width,
        'consensus_score': collapse.consensus_score,
        'dominant_driver': collapse.dominant_driver,
        'branch_count': len(branches),
        'top_branch_probability': branches[0]['probability'] if branches else 0.0,
        'ui_outputs': {
            'dashboard': str(FINAL_DASHBOARD_HTML_PATH),
            'cone': str(PROBABILITY_CONE_HTML_PATH),
            'persona': str(PERSONA_BREAKDOWN_HTML_PATH),
        },
    }
    LATEST_MARKET_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_MARKET_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

