from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def build_context_hash(context: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(context), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JudgeCacheEntry:
    context_hash: str
    features: dict[str, Any]
    decision: dict[str, Any]
    confidence: float
    risk_adjustment: float
    timestamp: str


class LocalJudgeCache:
    """
    Local decision cache for Claude/Kimi execution verdicts.
    Reuse decision when similarity > 0.92.
    """

    def __init__(self, path: Path, similarity_threshold: float = 0.92):
        self.path = Path(path)
        self.similarity_threshold = float(similarity_threshold)
        self.entries: list[JudgeCacheEntry] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    @staticmethod
    def _feature_vector(features: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                _safe_float(features.get("regime_confidence"), 0.0),
                _safe_float(features.get("admission_score"), 0.0),
                _safe_float(features.get("calibrated_probability"), 0.0),
                _safe_float(features.get("expected_rr"), 0.0),
                _safe_float(features.get("spread"), 0.0),
                _safe_float(features.get("slippage_estimate"), 0.0),
                _safe_float((features.get("recent_trade_health") or {}).get("rolling_win_rate_10"), 0.0),
                _safe_float((features.get("recent_trade_health") or {}).get("recent_drawdown"), 0.0),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _categorical_bonus(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        score = 0.0
        if str(left.get("regime", "")).lower() == str(right.get("regime", "")).lower():
            score += 0.04
        if str(left.get("strategic_direction", "")).upper() == str(right.get("strategic_direction", "")).upper():
            score += 0.04
        return score

    def similarity(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        a = self._feature_vector(left)
        b = self._feature_vector(right)
        distance = np.linalg.norm(a - b)
        normalized = 1.0 / (1.0 + distance)
        return float(np.clip(normalized + self._categorical_bonus(left, right), 0.0, 1.0))

    def lookup(self, features: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.entries:
            return None
        scored: list[tuple[float, JudgeCacheEntry]] = []
        for entry in self.entries:
            score = self.similarity(features, entry.features)
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_entry = scored[0]
        if best_score < self.similarity_threshold:
            return None
        return {
            "similarity": float(best_score),
            "decision": dict(best_entry.decision),
            "confidence": float(best_entry.confidence),
            "risk_adjustment": float(best_entry.risk_adjustment),
            "context_hash": best_entry.context_hash,
            "source": "local_judge_cache",
            "cached_at": best_entry.timestamp,
        }

    def add(self, features: Mapping[str, Any], decision: Mapping[str, Any]) -> JudgeCacheEntry:
        confidence = _safe_float(decision.get("confidence"), 0.0)
        risk_multiplier = _safe_float(decision.get("risk_multiplier", decision.get("size_multiplier", 0.7)), 0.7)
        risk_level = str(decision.get("risk_level", "HIGH")).upper()
        level_adjustment = {"LOW": 1.0, "MEDIUM": 0.85, "HIGH": 0.7}.get(risk_level, 0.7)
        risk_adjustment = float(np.clip(risk_multiplier * level_adjustment, 0.25, 1.0))
        entry = JudgeCacheEntry(
            context_hash=build_context_hash(features),
            features=dict(features),
            decision=dict(decision),
            confidence=confidence,
            risk_adjustment=float(risk_adjustment),
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        self.entries.append(entry)
        self._truncate(max_rows=5000)
        return entry

    def _truncate(self, max_rows: int) -> None:
        if len(self.entries) <= int(max_rows):
            return
        self.entries = self.entries[-int(max_rows) :]

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), ensure_ascii=True) + "\n")
        return self.path

    def load(self) -> None:
        self.entries = []
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                entry = JudgeCacheEntry(
                    context_hash=str(payload.get("context_hash", "")),
                    features=dict(payload.get("features", {})),
                    decision=dict(payload.get("decision", {})),
                    confidence=_safe_float(payload.get("confidence"), 0.0),
                    risk_adjustment=_safe_float(payload.get("risk_adjustment"), 1.0),
                    timestamp=str(payload.get("timestamp", "")),
                )
                self.entries.append(entry)
            except Exception:
                continue

    def build_from_logs(self, log_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        added = 0
        for row in log_rows:
            candidate = row.get("candidate", {})
            decision = row.get("decision", {})
            if not isinstance(candidate, Mapping) or not isinstance(decision, Mapping):
                continue
            if not decision.get("available", False):
                continue
            self.add(candidate, decision)
            added += 1
        self.save()
        return {"cache_entries": len(self.entries), "new_entries": int(added), "similarity_threshold": self.similarity_threshold}
