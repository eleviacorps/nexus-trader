from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Mapping

from config.project_config import OUTPUTS_DIR


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else (default or {})
    except Exception:
        return default or {}


@dataclass
class ProductionDashboard:
    outputs_root: Path = OUTPUTS_DIR

    @property
    def control_state_path(self) -> Path:
        return self.outputs_root / "live" / "v25_control_state.json"

    def get_control_state(self) -> dict[str, Any]:
        default = {
            "mode": "manual_mode",
            "trading_paused": False,
            "emergency_stop": False,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        return _read_json(self.control_state_path, default=default)

    def set_control_state(
        self,
        *,
        mode: str | None = None,
        trading_paused: bool | None = None,
        emergency_stop: bool | None = None,
    ) -> dict[str, Any]:
        state = self.get_control_state()
        if mode is not None:
            normalized = str(mode).strip().lower()
            if normalized not in {"manual_mode", "auto_mode"}:
                raise ValueError("mode must be manual_mode or auto_mode")
            state["mode"] = normalized
        if trading_paused is not None:
            state["trading_paused"] = bool(trading_paused)
        if emergency_stop is not None:
            state["emergency_stop"] = bool(emergency_stop)
            if bool(emergency_stop):
                state["trading_paused"] = True
                state["mode"] = "manual_mode"
        state["updated_at"] = datetime.now(tz=UTC).isoformat()
        self.control_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.control_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    def snapshot(self, dashboard_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        dashboard_payload = dict(dashboard_payload or {})
        deployment = _read_json(self.outputs_root / "deployment" / "deployment_readiness.json")
        final_validation = _read_json(self.outputs_root / "v25" / "final_validation.json")
        branch_eval = _read_json(self.outputs_root / "v25" / "branch_accuracy_evaluation.json")
        tradeability = _read_json(self.outputs_root / "v25" / "tradeability_training_report.json")
        cache_report = _read_json(self.outputs_root / "v25" / "local_judge_cache_report.json")
        control = self.get_control_state()

        simulation = dict(dashboard_payload.get("simulation", {}))
        consensus = simulation.get("consensus_path", [])
        minority = simulation.get("minority_path", [])
        current_regime = simulation.get("detected_regime", "unknown")
        current_trades = dict(dashboard_payload.get("paper_trading", {})).get("summary", {})

        score = _safe_float((deployment.get("score_breakdown") or {}).get("total_score_100"), 0.0)
        status = str(deployment.get("deployment_status", "BLOCKED"))
        return {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "system_status": status,
            "deployment_score": score,
            "current_regime": current_regime,
            "consensus_branch": consensus[:8] if isinstance(consensus, list) else [],
            "minority_branch": minority[:8] if isinstance(minority, list) else [],
            "current_trades": current_trades,
            "recent_performance": (final_validation.get("aggregate_metrics") or deployment.get("aggregate_metrics") or {}),
            "claude_decision": (dashboard_payload.get("kimi_judge", {}) or {}).get("content", {}),
            "tradeability_probability": _safe_float((tradeability.get("evaluation") or {}).get("mean_probability"), 0.0),
            "branch_realism_improvement": _safe_float(branch_eval.get("branch_realism_improvement_ratio"), 0.0),
            "tradeability_precision": _safe_float((tradeability.get("evaluation") or {}).get("precision_at_threshold"), 0.0),
            "cache_hit_rate": _safe_float(cache_report.get("cache_hit_rate"), 0.0),
            "control_state": control,
            "service_status": {
                "api": "running",
                "trader": "running" if not bool(control.get("trading_paused")) else "paused",
                "monitor": "running",
                "backup": "running",
            },
        }

