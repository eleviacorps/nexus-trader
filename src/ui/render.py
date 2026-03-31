from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from config.project_config import FINAL_DASHBOARD_HTML_PATH, PERSONA_BREAKDOWN_HTML_PATH, PROBABILITY_CONE_HTML_PATH

try:
    import plotly.graph_objects as go  # type: ignore
except ImportError:  # pragma: no cover
    go = None


def _require_plotly():
    if go is None:
        raise ImportError('plotly is required to render the Nexus dashboard.')


def render_persona_breakdown(persona_snapshot: Mapping[str, float], output_path: Path = PERSONA_BREAKDOWN_HTML_PATH) -> Path:
    _require_plotly()
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(persona_snapshot.keys()),
                y=[float(value) for value in persona_snapshot.values()],
                marker_color=['#c0392b', '#2c3e50', '#2980b9', '#16a085', '#8e44ad'][: len(persona_snapshot)],
            )
        ]
    )
    fig.update_layout(title='Persona Impact Breakdown', xaxis_title='Persona', yaxis_title='Impact', template='plotly_white')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    return output_path


def render_probability_dashboard(
    price_frame,
    branches: Sequence[Mapping[str, object]],
    cone_points: Sequence[Mapping[str, float]],
    consensus_score: float,
    dominant_driver: str,
    persona_snapshot: Mapping[str, float],
    output_path: Path = FINAL_DASHBOARD_HTML_PATH,
) -> Path:
    _require_plotly()
    frame = price_frame.copy().tail(240)
    if 'open' not in frame.columns:
        frame['open'] = frame['close']
    if 'high' not in frame.columns:
        frame['high'] = frame[['open', 'close']].max(axis=1)
    if 'low' not in frame.columns:
        frame['low'] = frame[['open', 'close']].min(axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame.index.astype(str),
            open=frame['open'],
            high=frame['high'],
            low=frame['low'],
            close=frame['close'],
            name='Price',
        )
    )

    if branches:
        for index, branch in enumerate(branches[:8]):
            fig.add_trace(
                go.Scatter(
                    x=branch['timestamps'],
                    y=branch['predicted_prices'],
                    mode='lines',
                    line={'width': 1, 'color': 'rgba(52, 152, 219, 0.18)'},
                    name=f"branch_{branch['path_id']}",
                    showlegend=False,
                    hovertemplate='Branch %{text}<br>Price=%{y:.2f}<extra></extra>',
                    text=[branch['path_id']] * len(branch['timestamps']),
                )
            )

    if cone_points:
        future_times = [point['timestamp'] for point in cone_points]
        lower = [point['lower_price'] for point in cone_points]
        center = [point['center_price'] for point in cone_points]
        upper = [point['upper_price'] for point in cone_points]
        fig.add_trace(go.Scatter(x=future_times, y=lower, mode='lines', line={'color': 'rgba(39, 174, 96, 0.0)'}, name='Lower', showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=future_times,
                y=upper,
                mode='lines',
                line={'color': 'rgba(39, 174, 96, 0.0)'},
                fill='tonexty',
                fillcolor='rgba(39, 174, 96, 0.15)',
                name='Probability Cone',
            )
        )
        fig.add_trace(go.Scatter(x=future_times, y=center, mode='lines+markers', line={'color': '#27ae60', 'width': 3}, name='Cone Center'))

    persona_text = '<br>'.join(f'{name}: {value:.3f}' for name, value in persona_snapshot.items()) if persona_snapshot else 'n/a'
    fig.update_layout(
        title=f'Nexus Trader Dashboard | consensus={consensus_score:.3f} | driver={dominant_driver}',
        template='plotly_white',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        annotations=[
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': 1.01,
                'y': 1.0,
                'showarrow': False,
                'align': 'left',
                'text': f'<b>Persona Snapshot</b><br>{persona_text}',
                'bordercolor': '#d5dbdb',
                'borderwidth': 1,
                'bgcolor': 'rgba(255,255,255,0.8)',
            }
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
    PROBABILITY_CONE_HTML_PATH.write_text(output_path.read_text(encoding='utf-8'), encoding='utf-8')
    return output_path


def write_branches_json(branches: Sequence[Mapping[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(branches), indent=2), encoding='utf-8')
    return path
