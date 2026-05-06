"""
Scorecard scaling — converts a Probability of Default (PD) into an integer
credit score on the 0–1000 scale.

Convention (higher = better credit, lower risk):
    score = round(1000 * (1 - pd))

This linear mapping is intentionally simple and transparent, making it easy
to defend in interviews. The white paper discusses non-linear alternatives.

Risk tiers:
    800–1000  Very Low      (PD < 20%)
    650–799   Low           (PD 20–35%)
    500–649   Medium        (PD 35–50%)
    300–499   High          (PD 50–70%)
      0–299   Very High     (PD > 70%)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


RISK_TIERS = [
    (800, "very_low"),
    (650, "low"),
    (500, "medium"),
    (300, "high"),
    (0,   "very_high"),
]


def pd_to_score(probability_of_default: float) -> int:
    """Convert PD in [0, 1] to an integer credit score in [0, 1000]."""
    pd_clipped = float(np.clip(probability_of_default, 0.0, 1.0))
    return round(1000 * (1.0 - pd_clipped))


def score_to_risk_tier(score: int) -> str:
    """Map a numeric score to a risk tier label."""
    for threshold, tier in RISK_TIERS:
        if score >= threshold:
            return tier
    return "very_high"


@dataclass
class ScoreResult:
    wallet_address: str
    probability_of_default: float
    score: int
    risk_tier: str
    top_factors: list[dict] = field(default_factory=list)
    model_version: str = "v1"

    @classmethod
    def from_pd(
        cls,
        wallet_address: str,
        pd: float,
        top_factors: list[dict] | None = None,
        model_version: str = "v1",
    ) -> "ScoreResult":
        score = pd_to_score(pd)
        tier = score_to_risk_tier(score)
        return cls(
            wallet_address=wallet_address,
            probability_of_default=pd,
            score=score,
            risk_tier=tier,
            top_factors=top_factors or [],
            model_version=model_version,
        )

    def to_dict(self) -> dict:
        return {
            "wallet_address": self.wallet_address,
            "score": self.score,
            "risk_tier": self.risk_tier,
            "probability_of_default": round(self.probability_of_default, 4),
            "top_factors": self.top_factors,
            "model_version": self.model_version,
        }
