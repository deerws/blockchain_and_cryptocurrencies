"""
Pydantic schemas for the ChainScore REST API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Single-wallet scoring ──────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    wallet_address: str = Field(
        ...,
        description="Ethereum wallet address (checksummed or lowercase hex)",
        examples=["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
    )
    include_shap: bool = Field(
        default=True,
        description="Include SHAP top-factor explanations in the response",
    )

    @field_validator("wallet_address")
    @classmethod
    def validate_ethereum_address(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError("wallet_address must be a 42-character hex string starting with 0x")
        return v


class ShapFactor(BaseModel):
    feature: str = Field(description="Feature name (snake_case)")
    shap_value: float = Field(description="SHAP contribution value (positive = increases risk)")
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    wallet_address: str
    score: int = Field(..., ge=0, le=1000, description="ChainScore (0 = highest risk, 1000 = lowest risk)")
    risk_tier: Literal["very_low", "low", "medium", "high", "very_high"] = Field(
        description="Risk classification: very_low (800–1000), low (650–799), medium (500–649), high (300–499), very_high (0–299)"
    )
    probability_of_default: float = Field(..., ge=0.0, le=1.0, description="Calibrated PD in [0, 1]")
    top_factors: list[ShapFactor] = Field(default_factory=list, description="Top SHAP factors (empty if include_shap=false)")
    model_version: str = Field(description="Model artifact version used for scoring")
    scored_at: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of scoring")
    score_valid_until: str = Field(description="Score validity date (30 days from scoring, YYYY-MM-DD)")


# ── Batch scoring ──────────────────────────────────────────────────────────

class BatchScoreRequest(BaseModel):
    wallet_addresses: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of Ethereum wallet addresses to score (max 20)",
        examples=[["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                   "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B"]],
    )
    include_shap: bool = Field(
        default=False,
        description="Include SHAP factors per wallet (slower — disabled by default for batch)",
    )

    @field_validator("wallet_addresses")
    @classmethod
    def validate_addresses(cls, addresses: list[str]) -> list[str]:
        cleaned = []
        for addr in addresses:
            addr = addr.strip()
            if not addr.startswith("0x") or len(addr) != 42:
                raise ValueError(f"Invalid Ethereum address: {addr!r}")
            cleaned.append(addr)
        if len(cleaned) != len(set(a.lower() for a in cleaned)):
            raise ValueError("Duplicate addresses are not allowed in a batch request")
        return cleaned


class BatchWalletResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    wallet_address: str
    score: int | None = Field(default=None, description="ChainScore, or null if scoring failed")
    risk_tier: str | None = Field(default=None)
    probability_of_default: float | None = Field(default=None)
    top_factors: list[ShapFactor] = Field(default_factory=list)
    error: str | None = Field(default=None, description="Error message if this wallet could not be scored")


class BatchScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    results: list[BatchWalletResult] = Field(description="One result per input wallet, in the same order")
    total: int = Field(description="Total wallets requested")
    succeeded: int = Field(description="Wallets successfully scored")
    failed: int = Field(description="Wallets that could not be scored")
    model_version: str = Field(description="Model version used")
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ── System ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    ethereum_connected: bool
