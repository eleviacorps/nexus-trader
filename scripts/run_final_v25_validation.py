from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_DIR


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else (default or {})
    except Exception:
        return default or {}


def _attempt_proxy_replay() -> dict[str, Any]:
    script_path = PROJECT_ROOT / "scripts" / "run_v25_proxy_paper.py"
    if not script_path.exists():
        return {"executed": False, "error": "missing_run_v25_proxy_paper_script"}
    python_exec = os.getenv("NEXUS_PYTHON", sys.executable)
    command = [python_exec, str(script_path)]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
    return {
        "executed": completed.returncode == 0,
        "return_code": int(completed.returncode),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]) if completed.stderr else "",
    }


def _run_script_if_exists(script_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "scripts" / script_name
    if not path.exists():
        return {"executed": False, "script": script_name, "error": "missing_script"}
    python_exec = os.getenv("NEXUS_PYTHON", sys.executable)
    command = [python_exec, str(path)]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
    return {
        "executed": completed.returncode == 0,
        "return_code": int(completed.returncode),
        "script": script_name,
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]) if completed.stderr else "",
    }


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    outputs_v25 = OUTPUTS_DIR / "v25"
    outputs_v25.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "live").mkdir(parents=True, exist_ok=True)

    recovery_run = _run_script_if_exists("run_v25_1_recovery.py")
    proxy_run = _attempt_proxy_replay()
    comparison_run = _run_script_if_exists("build_v25_final_metric_comparison.py")
    v25_validation = _load_json(outputs_v25 / "v25_1_validation.json")
    deployment = _load_json(OUTPUTS_DIR / "deployment" / "deployment_readiness.json")
    branch = _load_json(outputs_v25 / "branch_realism_report.json")
    tradeability = _load_json(outputs_v25 / "tradeability_training_report.json")
    live_proxy = _load_json(OUTPUTS_DIR / "live" / "live_paper_report.json")
    comparison = _load_json(outputs_v25 / "final_metric_comparison.json")

    aggregate = dict(v25_validation.get("aggregate_metrics", {}))
    participation = _safe_float(aggregate.get("participation_rate"), 0.0)
    win_rate = _safe_float(aggregate.get("win_rate"), 0.0)
    expectancy_r = _safe_float(aggregate.get("expectancy_R"), 0.0)
    drawdown = _safe_float(aggregate.get("max_drawdown"), 0.0)
    branch_improvement = _safe_float(v25_validation.get("branch_realism_improvement_ratio", branch.get("branch_realism_improvement_ratio")), 0.0)
    tradeability_precision = _safe_float((tradeability.get("evaluation") or {}).get("precision_at_threshold"), 0.0)
    deployment_score = _safe_float(v25_validation.get("deployment_readiness_score", (deployment.get("score_breakdown") or {}).get("total_score_100")), 0.0)
    proxy_positive = bool(live_proxy.get("proxy_positive", False))

    gates = {
        "participation_15_25": 0.15 <= participation <= 0.25,
        "win_rate_gt_60pct": win_rate > 0.60,
        "expectancy_gt_0_12R": expectancy_r > 0.12,
        "drawdown_lt_10pct": drawdown < 0.10,
        "branch_realism_improvement_gt_15pct": branch_improvement > 0.15,
        "tradeability_precision_gt_65pct": tradeability_precision > 0.65,
        "deployment_readiness_score_gt_90": deployment_score > 90.0,
        "proxy_14d_positive": proxy_positive,
    }
    final_ready = all(bool(value) for value in gates.values())
    deployment_status = "PRODUCTION READY" if final_ready else "BLOCKED"

    result = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "v25_1_recovery_run": recovery_run,
        "final_metric_comparison_run": comparison_run,
        "proxy_replay_run": proxy_run,
        "final_metric_comparison": comparison,
        "metrics": {
            "participation_rate": participation,
            "win_rate": win_rate,
            "expectancy_R": expectancy_r,
            "max_drawdown": drawdown,
            "branch_realism_improvement_ratio": branch_improvement,
            "tradeability_precision": tradeability_precision,
            "deployment_readiness_score": deployment_score,
            "proxy_positive": proxy_positive,
        },
        "targets": {
            "participation_range": [0.15, 0.25],
            "win_rate_min": 0.60,
            "expectancy_R_min": 0.12,
            "drawdown_max": 0.10,
            "branch_realism_improvement_ratio_min": 0.15,
            "tradeability_precision_min": 0.65,
            "deployment_readiness_score_min": 90.0,
            "proxy_positive_required": True,
        },
        "gates": gates,
        "deployment_status": deployment_status,
    }
    final_json_path = outputs_v25 / "final_validation.json"
    final_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# V25 Final Validation",
        "",
        f"Generated at: `{result['generated_at']}`",
        f"Deployment status: `{deployment_status}`",
        "",
        "## Metrics",
        f"- Participation: `{participation:.6f}` (target `0.15-0.25`)",
        f"- Win rate: `{win_rate:.6f}` (target `>0.60`)",
        f"- Expectancy (R): `{expectancy_r:.6f}` (target `>0.12`)",
        f"- Max drawdown: `{drawdown:.6f}` (target `<0.10`)",
        f"- Branch realism improvement: `{branch_improvement:.6f}` (target `>0.15`)",
        f"- Tradeability precision: `{tradeability_precision:.6f}` (target `>0.65`)",
        f"- Deployment readiness score: `{deployment_score:.6f}` (target `>90`)",
        f"- Proxy 14-day replay positive: `{proxy_positive}`",
        "",
        "## Gate Results",
    ]
    for key, passed in gates.items():
        lines.append(f"- `{key}`: `{'PASS' if passed else 'FAIL'}`")
    final_md_path = outputs_v25 / "final_validation.md"
    _write_markdown(final_md_path, lines)

    if final_ready:
        production_ready_path = outputs_v25 / "production_ready.json"
        guide_path = outputs_v25 / "deployment_guide.md"
        notes_path = outputs_v25 / "final_release_notes.md"
        blockers_path = outputs_v25 / "final_blockers.md"
        if blockers_path.exists():
            blockers_path.unlink()
        production_ready_path.write_text(
            json.dumps(
                {
                    "generated_at": result["generated_at"],
                    "deployment_status": deployment_status,
                    "readiness_score": deployment_score,
                    "manual_mode_burn_in_hours": 48,
                    "auto_mode_policy": {
                        "risk_per_trade": 0.0025,
                        "max_simultaneous_trades": 3,
                        "sell_stricter_than_buy": True,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_markdown(
            guide_path,
            [
                "# V25 Deployment Guide",
                "",
                "1. Start services with `python scripts/start_production.py --service all --loop` or docker compose.",
                "2. Keep `manual_mode` for the first 48 hours while monitoring logs and drawdown.",
                "3. Enable `auto_mode` only if readiness remains above 90 and all safety gates remain green.",
                "4. Keep risk controls at 0.25% risk, max 3 simultaneous trades, no pyramiding.",
            ],
        )
        _write_markdown(
            notes_path,
            [
                "# V25 Final Release Notes",
                "",
                "- V25.1 branch realism retraining and regime-specific branch ranking are active.",
                "- Adaptive expectancy gate and trade-cluster filtering are active.",
                "- GLM-based judge path and local fallback routing are active.",
                "- Production hosting and dashboard controls are wired.",
                "- Deployment status: PRODUCTION READY.",
            ],
        )
    else:
        failed = [name for name, passed in gates.items() if not passed]
        blockers_path = outputs_v25 / "final_blockers.md"
        blocker_lines = [
            "# V25 Final Blockers",
            "",
            f"Generated at: `{result['generated_at']}`",
            f"Deployment status: `{deployment_status}`",
            "",
            "Remaining failed gates with root cause and next fix:",
        ]
        root_causes = {
            "participation_15_25": ("Admission thresholds too tight in dominant regimes.", "Reduce trend_up threshold slightly while keeping adaptive loss guard."),
            "win_rate_gt_60pct": ("Trend-up accepted set still includes noisy entries.", "Increase trend-up quality threshold or add stronger trend-up branch-quality veto."),
            "expectancy_gt_0_12R": ("Low-quality clusters still admitted in difficult segments.", "Strengthen adaptive threshold increase after losses and tighten duplicate cluster radius."),
            "drawdown_lt_10pct": ("Risk concentration in clustered entries.", "Further limit same-regime rapid re-entry and apply stronger post-loss cooldown."),
            "branch_realism_improvement_gt_15pct": ("Current sequence realism model underfits regime-specific structure.", "Retrain per-regime rankers with richer path-shape features and recalibrate blend inputs."),
            "tradeability_precision_gt_65pct": ("Meta-model drift or poor thresholding.", "Recalibrate threshold and retrain on newest difficult windows."),
            "deployment_readiness_score_gt_90": ("Multiple metric gates still failed.", "Resolve win-rate/expectancy/branch-realism gates first, then rescore."),
            "proxy_14d_positive": ("Replay route still not converting approved candidates to profitable closures.", "Enable manual-confirmation replay execution path and reduce fallback size shock."),
        }
        for gate in failed:
            cause, next_fix = root_causes.get(gate, ("Metric below target.", "Tune corresponding model and rerun validation."))
            blocker_lines.append(f"- `{gate}`")
            blocker_lines.append(f"  - likely root cause: {cause}")
            blocker_lines.append(f"  - exact next fix: {next_fix}")
        _write_markdown(blockers_path, blocker_lines)

    print(
        json.dumps(
            {
                "final_validation_json": str(final_json_path),
                "final_validation_md": str(final_md_path),
                "deployment_status": deployment_status,
                "failed_gates": [name for name, passed in gates.items() if not passed],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
