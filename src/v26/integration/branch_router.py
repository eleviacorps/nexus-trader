"""
V26 Phase 1, Agent 5: V25 Integration Layer (SANDBOXED)

This module implements the integration layer between V26 regime-threaded branches
and the existing V25 execution pipeline. All execution is SANDBOXED until
evaluation passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
import json
import numpy as np
import torch
from torch import Tensor

from src.v24.world_state import WorldState
from src.v25.branch_quality_model import BranchQualityModel


# V6 Regime labels to V24.2 tactical labels mapping
V6_TO_V24_2_MAPPING = {
    "trend_up_strong": "trend_up",
    "trend_up_weak": "trend_up",
    "trend_down_weak": "trend_down",
    "trend_down_strong": "trend_down",
    "range": "range",
    "breakout": "breakout",
    "panic_news_shock": "panic",
    "mean_reversion": "range",  # or 'chop'
    "low_volatility": "range",
}


class NotReadyError(Exception):
    """Raised when V26 integration is not ready for execution."""
    pass


@dataclass(frozen=True)
class ExecutionDecision:
    """Execution decision output from the branch router."""
    branches: list[dict[str, Any]]
    recommended_regime: str
    confidence: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    sandboxed: bool = True  # Always True until evaluation passes


@dataclass
class RegimeThreadingResult:
    """Result from regime-threaded path generation."""
    paths: list[dict[str, Any]]
    top_3_regimes: list[str]
    regime_distribution: dict[str, float]


class RegimeThreadingController:
    """
    Placeholder for V26 Regime Threading Controller.
    This will be fully implemented in V26 Phase 1, Agents 1-4.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_regime_threaded_paths(
        self,
        world_state: WorldState,
        regime_probs: Tensor,
    ) -> RegimeThreadingResult:
        """
        Generate paths threaded through top-3 regime latent spaces.

        Args:
            world_state: Current market world state
            regime_probs: Regime probability distribution from V6 classifier

        Returns:
            RegimeThreadingResult containing paths and regime information
        """
        # Placeholder implementation - will be replaced by actual V26 implementation
        # For now, return synthetic paths for integration testing

        probs_np = regime_probs.detach().cpu().numpy() if isinstance(regime_probs, Tensor) else np.array(regime_probs)

        # Get top 3 regimes
        top_3_indices = np.argsort(probs_np)[-3:][::-1]
        regime_labels = list(V6_TO_V24_2_MAPPING.keys())

        top_3_regimes = [
            regime_labels[i] if i < len(regime_labels) else "range"
            for i in top_3_indices
        ]

        regime_distribution = {
            regime_labels[i]: float(probs_np[i])
            for i in range(min(len(regime_labels), len(probs_np)))
        }

        # Generate placeholder paths (will be replaced by actual V26 path generation)
        paths = []
        for i, regime in enumerate(top_3_regimes[:3]):
            path = {
                "path_id": f"v26_path_{i}",
                "regime": regime,
                "branch_volatility": 0.15 + i * 0.05,
                "branch_acceleration": 0.02 + i * 0.01,
                "regime_consistency": float(probs_np[top_3_indices[i]]),
                "analog_similarity": 0.7 - i * 0.1,
                "specialist_bot_agreement": 0.6 + i * 0.1,
                "cabr_score": 0.65 + i * 0.05,
                "minority_disagreement": 0.3 - i * 0.05,
                "historical_outcome_fit": 0.55 + i * 0.05,
                "path": [world_state.market_structure.get("close", 0.0)] * 10,  # Placeholder
            }
            paths.append(path)

        return RegimeThreadingResult(
            paths=paths,
            top_3_regimes=top_3_regimes,
            regime_distribution=regime_distribution,
        )


class RegimeSpecificBranchRanker:
    """
    Regime-specific branch ranking component.
    Applies regime-specific adjustments to branch rankings.
    """

    REGIME_WEIGHTS = {
        "trend_up": {"momentum": 1.2, "mean_reversion": 0.8, "breakout": 1.0},
        "trend_down": {"momentum": 1.2, "mean_reversion": 0.8, "breakout": 1.0},
        "range": {"momentum": 0.7, "mean_reversion": 1.3, "breakout": 0.9},
        "breakout": {"momentum": 1.1, "mean_reversion": 0.6, "breakout": 1.3},
        "panic": {"momentum": 0.8, "mean_reversion": 1.1, "breakout": 1.0},
    }

    def __init__(self) -> None:
        self.default_weights = {"momentum": 1.0, "mean_reversion": 1.0, "breakout": 1.0}

    def rank_by_regime(
        self,
        ranked_branches: list[dict[str, Any]],
        v24_2_regime: str,
    ) -> list[dict[str, Any]]:
        """
        Apply regime-specific ranking adjustments.

        Args:
            ranked_branches: Branches already ranked by V25 CABR
            v24_2_regime: Mapped V24.2 regime label

        Returns:
            Regime-adjusted branch rankings
        """
        weights = self.REGIME_WEIGHTS.get(v24_2_regime, self.default_weights)

        adjusted = []
        for branch in ranked_branches:
            # Apply regime-specific weighting
            base_score = float(branch.get("blended_rank_score", 0.0))
            regime_consistency = float(branch.get("regime_consistency", 0.5))

            # Boost score based on regime alignment
            adjusted_score = base_score * (1.0 + 0.2 * regime_consistency)

            branch_copy = dict(branch)
            branch_copy["regime_adjusted_score"] = float(np.clip(adjusted_score, 0.0, 1.0))
            branch_copy["applied_regime"] = v24_2_regime
            adjusted.append(branch_copy)

        # Re-sort by adjusted score
        return sorted(adjusted, key=lambda b: b.get("regime_adjusted_score", 0.0), reverse=True)


class V26BranchRouter:
    """
    V26 Branch Router: Maps V26 regime-threaded branches to V25 execution pipeline.

    This class is SANDBOXED - it will not execute trades until evaluation passes.
    All integration attempts are logged but not executed.
    """

    EVALUATION_REPORT_PATH = Path("outputs/v26/evaluation_report.json")
    INTEGRATION_LOG_PATH = Path("outputs/v26/integration_attempts.jsonl")

    def __init__(
        self,
        threading_controller: RegimeThreadingController | None = None,
        v25_branch_ranker: BranchQualityModel | None = None,
        v25_regime_ranker: RegimeSpecificBranchRanker | None = None,
    ) -> None:
        """
        Initialize the V26 Branch Router.

        Args:
            threading_controller: V26 regime threading controller
            v25_branch_ranker: V25 branch quality model for ranking
            v25_regime_ranker: Regime-specific branch ranker
        """
        self.threading_controller = threading_controller or RegimeThreadingController()
        self.branch_ranker = v25_branch_ranker or BranchQualityModel()
        self.regime_ranker = v25_regime_ranker or RegimeSpecificBranchRanker()

        # Placeholder execution hooks (SANDBOXED)
        self._execution_mode_router: Any = None
        self._paper_trade_engine: Any = None

    def _map_regime(self, v6_regime: str) -> str:
        """
        Map V6 regime label to V24.2 tactical label.

        Args:
            v6_regime: V6 regime classification label

        Returns:
            V24.2 tactical regime label
        """
        return V6_TO_V24_2_MAPPING.get(v6_regime, "range")

    def _check_evaluation_status(self) -> dict[str, Any]:
        """
        Check if V26 evaluation has passed.

        Returns:
            Evaluation status dictionary
        """
        if not self.EVALUATION_REPORT_PATH.exists():
            return {"exists": False, "passed": False, "reason": "evaluation_report_not_found"}

        try:
            report = json.loads(self.EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
            status = report.get("status", "UNKNOWN")
            return {
                "exists": True,
                "passed": status == "PASS",
                "status": status,
                "metrics": report.get("metrics", {}),
            }
        except Exception as e:
            return {"exists": True, "passed": False, "reason": f"error_reading_report: {e}"}

    def _log_integration_attempt(
        self,
        world_state: WorldState,
        decision: ExecutionDecision,
        evaluation_status: dict[str, Any],
    ) -> None:
        """Log integration attempt to file."""
        self.INTEGRATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": world_state.symbol,
            "decision": {
                "recommended_regime": decision.recommended_regime,
                "num_branches": len(decision.branches),
                "top_branch_score": decision.branches[0].get("regime_adjusted_score", 0.0) if decision.branches else 0.0,
            },
            "evaluation_status": evaluation_status,
            "sandboxed": decision.sandboxed,
        }

        with open(self.INTEGRATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def route_branches(
        self,
        world_state: WorldState,
        regime_probs: Tensor,
    ) -> ExecutionDecision:
        """
        Route V26 regime-threaded branches through V25 execution pipeline.

        Args:
            world_state: Current market world state
            regime_probs: Regime probability distribution from V6 classifier

        Returns:
            ExecutionDecision with routed branches (SANDBOXED)

        Raises:
            NotReadyError: If evaluation has not passed
        """
        # Check evaluation status
        evaluation_status = self._check_evaluation_status()

        # Step 1: Generate regime-threaded paths
        result = self.threading_controller.generate_regime_threaded_paths(
            world_state, regime_probs
        )
        paths = result.paths

        # Step 2: Map v6 regimes to v24.2 tactical labels
        v24_2_regime = self._map_regime(result.top_3_regimes[0])

        # Step 3: Run through V25 CABR
        ranked_branches = self.branch_ranker.rank_branches(paths)

        # Step 4: Apply regime-specific ranking
        final_branches = self.regime_ranker.rank_by_regime(
            ranked_branches, v24_2_regime
        )

        # Step 5: Create execution decision (SANDBOXED)
        decision = ExecutionDecision(
            branches=final_branches,
            recommended_regime=v24_2_regime,
            confidence=result.regime_distribution,
            sandboxed=True,  # Always sandboxed until evaluation passes
        )

        # Log integration attempt
        self._log_integration_attempt(world_state, decision, evaluation_status)

        # If evaluation failed, raise NotReadyError
        if not evaluation_status.get("passed", False):
            raise NotReadyError(
                f"V26 integration not ready: {evaluation_status.get('reason', 'evaluation_not_passed')}"
            )

        return decision

    def get_execution_hooks(self) -> dict[str, Any]:
        """
        Get placeholder execution hooks.
        These are SANDBOXED and will not execute until evaluation passes.

        Returns:
            Dictionary with execution_mode_router and paper_trade_engine placeholders
        """
        return {
            "execution_mode_router": self._execution_mode_router,
            "paper_trade_engine": self._paper_trade_engine,
            "note": "SANDBOXED - execution hooks are placeholders until evaluation passes",
        }

    def set_execution_mode_router(self, router: Any) -> None:
        """Set the execution mode router (placeholder)."""
        self._execution_mode_router = router

    def set_paper_trade_engine(self, engine: Any) -> None:
        """Set the paper trade engine (placeholder)."""
        self._paper_trade_engine = engine


class V26IntegrationLayer:
    """
    Complete V26 Integration Layer with safety checks and sandboxing.
    """

    def __init__(
        self,
        branch_router: V26BranchRouter | None = None,
    ) -> None:
        """
        Initialize the V26 Integration Layer.

        Args:
            branch_router: V26 branch router instance
        """
        self.branch_router = branch_router or V26BranchRouter()
        self._initialized = False

    def initialize(self) -> dict[str, Any]:
        """
        Initialize the integration layer and verify safety checks.

        Returns:
            Initialization status dictionary
        """
        eval_status = self.branch_router._check_evaluation_status()

        self._initialized = True

        return {
            "initialized": True,
            "evaluation_passed": eval_status.get("passed", False),
            "evaluation_status": eval_status,
            "sandboxed": True,
            "mode": "SANDBOX" if not eval_status.get("passed") else "READY",
        }

    def process_signal(
        self,
        world_state: WorldState,
        regime_probs: Tensor,
    ) -> dict[str, Any]:
        """
        Process a trading signal through the V26 integration layer.

        Args:
            world_state: Current market world state
            regime_probs: Regime probability distribution

        Returns:
            Processing result with decision and metadata
        """
        if not self._initialized:
            self.initialize()

        try:
            decision = self.branch_router.route_branches(world_state, regime_probs)
            return {
                "success": True,
                "decision": {
                    "recommended_regime": decision.recommended_regime,
                    "confidence": decision.confidence,
                    "num_branches": len(decision.branches),
                    "top_branch": decision.branches[0] if decision.branches else None,
                    "sandboxed": decision.sandboxed,
                },
                "execution_hooks": self.branch_router.get_execution_hooks(),
            }
        except NotReadyError as e:
            return {
                "success": False,
                "error": str(e),
                "sandboxed": True,
                "execution_hooks": None,
            }


def create_v26_integration_layer() -> V26IntegrationLayer:
    """Factory function to create a V26 integration layer instance."""
    return V26IntegrationLayer()


__all__ = [
    "V26BranchRouter",
    "V26IntegrationLayer",
    "RegimeThreadingController",
    "RegimeSpecificBranchRanker",
    "ExecutionDecision",
    "NotReadyError",
    "V6_TO_V24_2_MAPPING",
    "create_v26_integration_layer",
]
