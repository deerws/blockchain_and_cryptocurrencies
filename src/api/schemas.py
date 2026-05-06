"""
Pydantic schemas for the ChainScore REST API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScoreRequest(BaseModel):
    wallet_address: str = Field(
        ...,
        description="Ethereum wallet address (checksummed or lowercase hex)",
        examples=["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
    )
    include_shap: bool = Field(
        default=True,
        description="Whether to include SHAP factor explanations in the response",
    )

    @field_validator("wallet_address")
    @classmethod
    def validate_ethereum_address(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError("wallet_address must be a 42-character hex string starting with 0x")
        return v


class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResponse(BaseModel):
    wallet_address: str
    score: int = Field(..., ge=0, le=1000, description="ChainScore (0=highest risk, 1000=lowest risk)")
    risk_tier: Literal["very_low", "low", "medium", "high", "very_high"]
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    top_factors: list[ShapFactor] = Field(default_factory=list)
    model_version: str
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    score_valid_until: str = Field(
        ..., description="Approximate validity period (30 days from scoring)"
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    ethereum_connected: bool
