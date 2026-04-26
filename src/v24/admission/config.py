"""
V24 Adaptive Admission Layer Configuration
=========================================

Configuration for the adaptive admission layer system.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class AdmissionConfig:
    """Configuration for the adaptive admission layer."""
    
    # Base admission thresholds
    min_admission_score: float = 0.6
    min_quality_score: float = 0.1
    max_participation_rate: float = 0.4
    
    # Risk-based adjustments
    high_volatility_participation_cap: float = 0.25
    medium_volatility_participation_cap: float = 0.35
    low_volatility_participation_cap: float = 0.45
    
    # Performance-based adjustments
    drawdown_protection_threshold: float = 0.05
    consecutive_loss_limit: int = 3
    
    # Regime sensitivity
    regime_confidence_threshold: float = 0.6
    high_volatility_threshold: float = 2.5
    low_volatility_threshold: float = 1.5
    
    # Weight multipliers for scoring components
    regime_fit_weight: float = 0.25
    quality_score_weight: float = 0.30
    confidence_weight: float = 0.20
    risk_adjustment_weight: float = 0.15
    performance_weight: float = 0.10


# Default configuration
DEFAULT_ADMISSION_CONFIG = AdmissionConfig()
