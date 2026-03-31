from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config.project_config import (
    CROWD_EMBEDDINGS_INDEX_PATH,
    CROWD_EMBEDDINGS_NPY_PATH,
    CROWD_EMBEDDINGS_PATH,
    CROWD_EVENTS_PATH,
    FEATURE_DIM_CROWD,
    FEATURE_DIM_NEWS,
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    NEWS_EMBEDDINGS_INDEX_PATH,
    NEWS_EMBEDDINGS_NPY_PATH,
    NEWS_EMBEDDINGS_PATH,
    NEWS_EMBEDDINGS_RAW_PATH,
    NEWS_EVENTS_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
)
from src.pipeline.fusion import load_price_frame, save_numpy_artifact

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
except ImportError:  # pragma: no cover
    TruncatedSVD = None
    HashingVectorizer = None


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for perception pipelines.")
    return pd


def _coerce_datetime(series):
    pandas = _require_pandas()
    values = pandas.to_datetime(series, errors="coerce", utc=True)
    if getattr(values.dt, "tz", None) is not None:
        values = values.dt.tz_convert(None)
    return values


def _parse_compact_datetime(series):
    pandas = _require_pandas()
    text = series.astype(str).str.strip()
    parsed = pandas.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(None)
    return parsed


def _normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    return frame


def _candidate_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        lowered_column = column.lower()
        if any(candidate.lower() in lowered_column for candidate in candidates):
            return column
    return None


def resolve_price_frame():
    price_path = next(
        (path for path in [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV] if path.exists()),
        None,
    )
    if price_path is None:
        raise FileNotFoundError("No price feature artifact found for timestamp alignment.")
    frame = load_price_frame(price_path)
    pandas = _require_pandas()
    frame = frame.copy()
    frame.index = pandas.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()]
    frame = frame.sort_index()
    return frame, price_path


def reduce_text_embeddings(texts: Sequence[str], output_dim: int, hash_dim: int = 512) -> np.ndarray:
    if not texts:
        return np.zeros((0, output_dim), dtype=np.float32)
    if HashingVectorizer is None or TruncatedSVD is None:
        raise ImportError("scikit-learn is required for text embedding reduction.")

    vectorizer = HashingVectorizer(
        n_features=max(hash_dim, output_dim),
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.transform(list(texts))
    sample_count = matrix.shape[0]
    dense = matrix.toarray().astype(np.float32)
    if sample_count > output_dim:
        svd = TruncatedSVD(n_components=output_dim, random_state=42)
        dense = svd.fit_transform(matrix).astype(np.float32)
    elif dense.shape[1] >= output_dim:
        dense = dense[:, :output_dim]

    if dense.shape[1] < output_dim:
        padded = np.zeros((dense.shape[0], output_dim), dtype=np.float32)
        padded[:, : dense.shape[1]] = dense
        dense = padded
    return dense.astype(np.float32, copy=False)


def align_event_matrix(price_index, event_frame, feature_columns: Sequence[str], tolerance_minutes: int):
    pandas = _require_pandas()
    if len(price_index) == 0:
        raise ValueError("Price index is empty.")

    bars = pandas.DataFrame({"timestamp": pandas.to_datetime(price_index, errors="coerce")}).dropna()
    bars["timestamp"] = bars["timestamp"].astype("datetime64[ns]")
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    if event_frame.empty:
        aligned = np.zeros((len(bars), len(feature_columns)), dtype=np.float32)
        index_frame = bars.copy()
        index_frame["source_timestamp"] = pandas.NaT
        return aligned, index_frame

    events = event_frame.copy()
    events["timestamp"] = pandas.to_datetime(events["timestamp"], errors="coerce").astype("datetime64[ns]")
    events = events.sort_values("timestamp").reset_index(drop=True)
    events["source_timestamp"] = events["timestamp"]
    merge_columns = ["timestamp", "source_timestamp", *feature_columns]
    if "source" in events.columns:
        merge_columns.append("source")
    merged = pandas.merge_asof(
        bars,
        events[merge_columns],
        on="timestamp",
        direction="backward",
        tolerance=pandas.Timedelta(minutes=tolerance_minutes),
    )
    aligned = merged[list(feature_columns)].fillna(0.0).to_numpy(dtype=np.float32)
    index_frame = merged[["timestamp", "source_timestamp"]].copy()
    if "source" in merged.columns:
        index_frame["source"] = merged["source"].fillna("")
    return aligned, index_frame


def save_frame(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv_relaxed(path: Path):
    pandas = _require_pandas()
    try:
        return pandas.read_csv(path)
    except Exception:
        return pandas.read_csv(path, engine="python")


def load_news_events(raw_news_dir: Path):
    pandas = _require_pandas()
    frames = []
    for path in sorted(raw_news_dir.rglob("*.csv")):
        frame = _read_csv_relaxed(path)
        frame = _normalize_columns(frame)
        timestamp_column = _candidate_column(frame.columns, ["seendate", "seen_date", "date", "timestamp"])
        title_column = _candidate_column(frame.columns, ["title", "headline"])
        if title_column is None:
            continue

        if timestamp_column is None:
            timestamps = pandas.Series([pandas.NaT] * len(frame))
        else:
            raw_timestamp = frame[timestamp_column]
            if raw_timestamp.astype(str).str.fullmatch(r"\d{14}").all():
                timestamps = _parse_compact_datetime(raw_timestamp)
            else:
                timestamps = _coerce_datetime(raw_timestamp)

        url_column = _candidate_column(frame.columns, ["url", "documentidentifier", "link"])
        source_column = _candidate_column(frame.columns, ["domain", "sourcecountry", "source", "sourcename"])
        text_values = frame[title_column].fillna("").astype(str).str.strip()
        output = pandas.DataFrame(
            {
                "timestamp": timestamps,
                "text": text_values,
                "source": frame[source_column].fillna(path.stem).astype(str) if source_column else path.stem,
                "url": frame[url_column].fillna("").astype(str) if url_column else "",
                "dataset": path.stem,
            }
        )
        output = output[(output["timestamp"].notna()) & (output["text"] != "")]
        if not output.empty:
            frames.append(output)

    if not frames:
        return pandas.DataFrame(columns=["timestamp", "text", "source", "url", "dataset"])

    combined = pandas.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "text"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def build_news_artifacts(raw_news_dir: Path, tolerance_minutes: int = 180):
    pandas = _require_pandas()
    price_frame, price_path = resolve_price_frame()
    events = load_news_events(raw_news_dir)
    texts = events["text"].tolist()
    vectors = reduce_text_embeddings(texts, FEATURE_DIM_NEWS)
    feature_columns = [f"news_{index:02d}" for index in range(FEATURE_DIM_NEWS)]

    if len(events) > 0:
        event_embeddings = pandas.DataFrame(vectors, columns=feature_columns)
        event_frame = pandas.concat([events.reset_index(drop=True), event_embeddings], axis=1)
    else:
        event_frame = events.copy()
        for column in feature_columns:
            event_frame[column] = []

    aligned, index_frame = align_event_matrix(price_frame.index, event_frame, feature_columns, tolerance_minutes=tolerance_minutes)

    save_frame(NEWS_EVENTS_PATH, events)
    save_frame(NEWS_EMBEDDINGS_PATH, event_frame)
    save_frame(NEWS_EMBEDDINGS_INDEX_PATH, index_frame)
    save_numpy_artifact(NEWS_EMBEDDINGS_RAW_PATH, aligned)
    save_numpy_artifact(NEWS_EMBEDDINGS_NPY_PATH, vectors)

    report = {
        "price_source": str(price_path),
        "raw_event_count": int(len(events)),
        "aligned_rows": int(aligned.shape[0]),
        "feature_dim": int(aligned.shape[1]) if aligned.ndim == 2 else 0,
        "coverage_ratio": float((aligned.any(axis=1).sum() / len(aligned)) if len(aligned) else 0.0),
        "sources": sorted(events["source"].astype(str).unique().tolist()) if len(events) else [],
    }
    return report


def load_crowd_events(raw_crowd_dir: Path):
    pandas = _require_pandas()
    frames = []
    sentiment_path = raw_crowd_dir / "sentiment" / "alternative_me_fng.json"
    if sentiment_path.exists():
        payload = json.loads(sentiment_path.read_text(encoding="utf-8"))
        rows = payload.get("data", [])
        frame = pandas.DataFrame(rows)
        if not frame.empty:
            timestamps = pandas.to_datetime(frame["timestamp"].astype("int64"), unit="s", utc=True).dt.tz_convert(None)
            parsed = pandas.DataFrame(
                {
                    "timestamp": timestamps,
                    "value": pandas.to_numeric(frame["value"], errors="coerce"),
                    "classification": frame.get("value_classification", "").astype(str),
                    "source": "alternative_me_fng",
                }
            )
            parsed = parsed.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
            frames.append(parsed)

    cftc_frames = []
    market_patterns = {
        "gold": "gold",
        "eurusd": "euro fx",
        "btc": "bitcoin",
    }
    for path in sorted(raw_crowd_dir.rglob("cftc/*.zip")):
        try:
            frame = pandas.read_csv(path, compression="zip", low_memory=False)
        except Exception:
            continue
        frame = _normalize_columns(frame)
        date_column = _candidate_column(frame.columns, ["report_date_as_yyyy-mm-dd", "report_date"])
        market_column = _candidate_column(frame.columns, ["market_and_exchange_names"])
        open_interest_column = _candidate_column(frame.columns, ["open_interest_all"])
        lev_long_column = _candidate_column(frame.columns, ["lev_money_positions_long_all"])
        lev_short_column = _candidate_column(frame.columns, ["lev_money_positions_short_all"])
        asset_long_column = _candidate_column(frame.columns, ["asset_mgr_positions_long_all"])
        asset_short_column = _candidate_column(frame.columns, ["asset_mgr_positions_short_all"])
        if None in [date_column, market_column, open_interest_column, lev_long_column, lev_short_column, asset_long_column, asset_short_column]:
            continue

        frame["timestamp"] = pandas.to_datetime(frame[date_column], errors="coerce")
        open_interest = pandas.to_numeric(frame[open_interest_column], errors="coerce").replace(0.0, np.nan)
        lev_net = (pandas.to_numeric(frame[lev_long_column], errors="coerce") - pandas.to_numeric(frame[lev_short_column], errors="coerce")) / open_interest
        asset_net = (pandas.to_numeric(frame[asset_long_column], errors="coerce") - pandas.to_numeric(frame[asset_short_column], errors="coerce")) / open_interest
        frame["net_signal"] = ((lev_net.fillna(0.0) + asset_net.fillna(0.0)) / 2.0).clip(-1.0, 1.0)

        for key, pattern in market_patterns.items():
            subset = frame[frame[market_column].astype(str).str.lower().str.contains(pattern, na=False)].copy()
            if subset.empty:
                continue
            subset = subset[["timestamp", "net_signal"]].dropna().sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            subset = subset.rename(columns={"net_signal": f"cftc_{key}_net"})
            cftc_frames.append(subset)

    for path in sorted(raw_crowd_dir.rglob("reddit/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        children = payload.get("data", {}).get("children", [])
        rows = []
        for child in children:
            data = child.get("data", {})
            title = str(data.get("title", "")).strip()
            selftext = str(data.get("selftext", "")).strip()
            text = " ".join(part for part in [title, selftext] if part)
            created = data.get("created_utc")
            if not text or created is None:
                continue
            rows.append(
                {
                    "timestamp": pandas.to_datetime(float(created), unit="s", utc=True).tz_convert(None),
                    "text": text,
                    "score": float(data.get("score", 0.0)),
                    "num_comments": float(data.get("num_comments", 0.0)),
                    "classification": path.stem,
                    "source": path.stem,
                }
            )
        if rows:
            frames.append(pandas.DataFrame(rows))

    if not frames:
        base = pandas.DataFrame(columns=["timestamp", "value", "classification", "source"])
    else:
        base = pandas.concat(frames, ignore_index=True)
        base = base.drop_duplicates(subset=["timestamp", "source", "classification"]).sort_values("timestamp").reset_index(drop=True)

    if cftc_frames:
        cftc = pandas.concat(cftc_frames, ignore_index=True)
        cftc = cftc.sort_values("timestamp").groupby("timestamp", as_index=False).last()
        if base.empty:
            base = cftc.copy()
            base["value"] = np.nan
            base["classification"] = "cftc"
            base["source"] = "cftc"
        else:
            base = base.merge(cftc, on="timestamp", how="outer")

    if base.empty:
        return pandas.DataFrame(columns=["timestamp", "value", "classification", "source"])

    for column in ["value", "cftc_gold_net", "cftc_eurusd_net", "cftc_btc_net"]:
        if column in base.columns:
            base[column] = pandas.to_numeric(base[column], errors="coerce").ffill()
    if "classification" in base.columns:
        base["classification"] = base["classification"].ffill().fillna("unknown")
    if "source" in base.columns:
        base["source"] = base["source"].ffill().fillna("composite_crowd")
    base = base.sort_values("timestamp").reset_index(drop=True)
    return base


def _classification_one_hot(values: Sequence[str], label: str) -> np.ndarray:
    return np.asarray([1.0 if str(value).strip().lower() == label else 0.0 for value in values], dtype=np.float32)


def build_crowd_numeric_vectors(frame, output_dim: int = FEATURE_DIM_CROWD) -> np.ndarray:
    if len(frame) == 0:
        return np.zeros((0, output_dim), dtype=np.float32)

    pandas = _require_pandas()
    working = frame.copy().sort_values("timestamp").reset_index(drop=True)
    def series_or_zeros(name: str):
        if name in working.columns:
            return pandas.to_numeric(working[name], errors="coerce").fillna(0.0).astype(float)
        return pandas.Series(np.zeros(len(working), dtype=np.float32))

    score = pandas.to_numeric(working.get("value", working.get("score", 0.0)), errors="coerce").fillna(0.0).astype(float)
    cftc_gold = series_or_zeros("cftc_gold_net")
    cftc_eurusd = series_or_zeros("cftc_eurusd_net")
    cftc_btc = series_or_zeros("cftc_btc_net")
    score_norm = (score / 100.0).clip(0.0, 1.0)
    delta_1 = score_norm.diff().fillna(0.0)
    delta_3 = score_norm.diff(3).fillna(0.0)
    delta_7 = score_norm.diff(7).fillna(0.0)
    mean_3 = score_norm.rolling(3, min_periods=1).mean()
    mean_7 = score_norm.rolling(7, min_periods=1).mean()
    mean_30 = score_norm.rolling(30, min_periods=1).mean()
    std_7 = score_norm.rolling(7, min_periods=1).std().fillna(0.0)
    std_30 = score_norm.rolling(30, min_periods=1).std().fillna(0.0)
    z_7 = ((score_norm - mean_7) / std_7.replace(0.0, np.nan)).fillna(0.0)
    z_30 = ((score_norm - mean_30) / std_30.replace(0.0, np.nan)).fillna(0.0)
    rolling_min_14 = score_norm.rolling(14, min_periods=1).min()
    rolling_max_14 = score_norm.rolling(14, min_periods=1).max()
    rolling_range_14 = (rolling_max_14 - rolling_min_14).fillna(0.0)
    cftc_gold_delta = cftc_gold.diff().fillna(0.0)
    cftc_eurusd_delta = cftc_eurusd.diff().fillna(0.0)
    cftc_btc_delta = cftc_btc.diff().fillna(0.0)
    cftc_mean = (cftc_gold + cftc_eurusd + cftc_btc) / 3.0
    cftc_dispersion = np.std(np.column_stack([cftc_gold, cftc_eurusd, cftc_btc]), axis=1)
    classification = working.get("classification", "").astype(str).str.strip().str.lower()
    timestamp = pandas.to_datetime(working["timestamp"], errors="coerce")
    dow = timestamp.dt.dayofweek.fillna(0).astype(float)
    month = timestamp.dt.month.fillna(1).astype(float)

    columns = [
        score_norm,
        delta_1,
        delta_3,
        delta_7,
        mean_3,
        mean_7,
        mean_30,
        std_7,
        std_30,
        z_7,
        z_30,
        (score_norm - rolling_min_14),
        (rolling_max_14 - score_norm),
        rolling_range_14,
        (score_norm >= 0.75).astype(float),
        (score_norm <= 0.25).astype(float),
        _classification_one_hot(classification, "extreme fear"),
        _classification_one_hot(classification, "fear"),
        _classification_one_hot(classification, "greed"),
        _classification_one_hot(classification, "extreme greed"),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
        cftc_gold,
        cftc_eurusd,
        cftc_btc,
        cftc_gold_delta,
        cftc_eurusd_delta,
        cftc_btc_delta,
        cftc_mean,
        cftc_dispersion,
    ]
    matrix = np.column_stack(columns).astype(np.float32)
    if matrix.shape[1] != output_dim:
        raise ValueError(f"Expected crowd feature width {output_dim}, got {matrix.shape[1]}")
    return matrix


def build_crowd_artifacts(raw_crowd_dir: Path, tolerance_minutes: int = 1440):
    pandas = _require_pandas()
    price_frame, price_path = resolve_price_frame()
    events = load_crowd_events(raw_crowd_dir)
    feature_columns = [f"crowd_{index:02d}" for index in range(FEATURE_DIM_CROWD)]
    vectors = build_crowd_numeric_vectors(events, output_dim=FEATURE_DIM_CROWD)

    if len(events) > 0:
        event_embeddings = pandas.DataFrame(vectors, columns=feature_columns)
        event_frame = pandas.concat([events.reset_index(drop=True), event_embeddings], axis=1)
    else:
        event_frame = events.copy()
        for column in feature_columns:
            event_frame[column] = []

    aligned, index_frame = align_event_matrix(price_frame.index, event_frame, feature_columns, tolerance_minutes=tolerance_minutes)

    save_frame(CROWD_EVENTS_PATH, events)
    save_frame(CROWD_EMBEDDINGS_PATH, event_frame)
    save_frame(CROWD_EMBEDDINGS_INDEX_PATH, index_frame)
    save_numpy_artifact(CROWD_EMBEDDINGS_NPY_PATH, aligned)

    report = {
        "price_source": str(price_path),
        "raw_event_count": int(len(events)),
        "aligned_rows": int(aligned.shape[0]),
        "feature_dim": int(aligned.shape[1]) if aligned.ndim == 2 else 0,
        "coverage_ratio": float((aligned.any(axis=1).sum() / len(aligned)) if len(aligned) else 0.0),
        "sources": sorted(events["source"].astype(str).unique().tolist()) if len(events) else [],
    }
    return report


def _collapse_yfinance_columns(frame):
    pandas = _require_pandas()
    if not isinstance(frame.columns, pandas.MultiIndex):
        return frame
    flat = []
    for column in frame.columns:
        parts = [str(part) for part in column if str(part) and not str(part).startswith("Unnamed")]
        flat.append("_".join(parts))
    frame = frame.copy()
    frame.columns = flat
    return frame


def _extract_series_from_table(path: Path):
    pandas = _require_pandas()
    if "yfinance" in path.parts:
        frame = None
        for header, index_col in (((0, 1), 0), (0, None)):
            try:
                frame = pandas.read_csv(path, header=header, index_col=index_col)
                break
            except Exception:
                frame = None
        if frame is None:
            raise ValueError(f"Unable to parse yfinance file {path}")
        frame = _collapse_yfinance_columns(frame)
        output_dates = pandas.to_datetime(frame.index, errors="coerce")
        if output_dates.notna().sum() == 0:
            date_column = _candidate_column(frame.columns, ["date"])
            if date_column is None:
                raise ValueError(f"Missing date column in {path.name}")
            output_dates = pandas.to_datetime(frame[date_column], errors="coerce")
        close_column = _candidate_column(frame.columns, ["adj_close", "adj close", "close"])
        if close_column is None:
            numeric_candidates = [column for column in frame.columns if pandas.api.types.is_numeric_dtype(frame[column])]
            if not numeric_candidates:
                raise ValueError(f"Missing close column in {path.name}")
            close_column = numeric_candidates[0]
        series = pandas.to_numeric(frame[close_column], errors="coerce")
        output = pandas.DataFrame({"date": output_dates, path.stem.replace("=", "_"): series})
        return output.dropna(subset=["date"])

    frame = pandas.read_csv(path)
    frame = _normalize_columns(frame)
    date_column = _candidate_column(frame.columns, ["date"])
    value_column = _candidate_column(frame.columns, [path.stem.lower(), "value", "close"])
    if date_column is None or value_column is None:
        raise ValueError(f"Unable to parse macro file {path.name}")
    output = pandas.DataFrame(
        {
            "date": pandas.to_datetime(frame[date_column], errors="coerce"),
            path.stem: pandas.to_numeric(frame[value_column], errors="coerce"),
        }
    )
    return output.dropna(subset=["date"])


def build_macro_artifacts(raw_macro_dir: Path):
    pandas = _require_pandas()
    frames = []
    for path in sorted(raw_macro_dir.rglob("*.csv")):
        try:
            frame = _extract_series_from_table(path)
        except Exception:
            continue
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No macro CSV files could be parsed.")

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="date", how="outer")
    combined = combined.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    value_columns = [column for column in combined.columns if column != "date"]
    combined[value_columns] = combined[value_columns].ffill()
    combined = combined.dropna(how="all", subset=value_columns)
    return combined
