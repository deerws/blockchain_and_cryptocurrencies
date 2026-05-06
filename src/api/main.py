"""
ChainScore FastAPI service — serves real-time credit scores for Ethereum wallets.

Endpoints:
    GET  /health        — liveness check
    POST /v1/score      — score a wallet address
    GET  /v1/score/{address} — same via GET (for quick browser testing)

Authentication: API key via X-API-Key header (configured in .env).

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, ScoreRequest, ScoreResponse, ShapFactor
from src.data.ethereum_client import EthereumClient
from src.models.predict import ChainScorePredictor

logger = logging.getLogger(__name__)

# ── Globals loaded at startup ──────────────────────────────────────────────
_client: EthereumClient | None = None
_predictor: ChainScorePredictor | None = None
_ethereum_connected: bool = False

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _check_api_key(key: str | None = Security(api_key_header)) -> str:
    secret = os.getenv("API_KEY_SECRET", "")
    if not secret:
        return "no-auth"  # Auth disabled when secret not configured
    if key != secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and establish Ethereum connection at startup."""
    global _client, _predictor, _ethereum_connected

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        _client = EthereumClient.from_env()
        _ = _client.get_latest_block()
        _ethereum_connected = True
        logger.info("Ethereum client connected.")
    except Exception as exc:
        logger.warning(f"Ethereum client failed to connect: {exc}")
        _ethereum_connected = False

    try:
        _predictor = ChainScorePredictor()
        _predictor._load()
    except FileNotFoundError as exc:
        logger.warning(f"Models not loaded: {exc}")
        _predictor = None

    yield

    logger.info("Shutting down ChainScore API.")


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChainScore API",
    description=(
        "On-chain credit scoring for DeFi wallets. "
        "Returns a 0–1000 credit score and Probability of Default (PD) "
        "based on Ethereum wallet behavioral analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _ethereum_connected and _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        ethereum_connected=_ethereum_connected,
    )


def _build_score_response(wallet: str, include_shap: bool) -> ScoreResponse:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Run training pipeline first.",
        )
    if _client is None or not _ethereum_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ethereum client not connected. Check ALCHEMY_API_KEY.",
        )

    result = _predictor.score_wallet(wallet, _client, use_shap=include_shap)
    valid_until = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")

    return ScoreResponse(
        wallet_address=result.wallet_address,
        score=result.score,
        risk_tier=result.risk_tier,
        probability_of_default=result.probability_of_default,
        top_factors=[
            ShapFactor(
                feature=f["feature"],
                shap_value=f["shap_value"],
                direction=f["direction"],
            )
            for f in result.top_factors
        ],
        model_version=result.model_version,
        score_valid_until=valid_until,
    )


@app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
async def score_wallet_post(
    request: ScoreRequest,
    _: str = Depends(_check_api_key),
) -> ScoreResponse:
    """Score an Ethereum wallet and return its ChainScore."""
    return _build_score_response(request.wallet_address, request.include_shap)


@app.get("/v1/score/{wallet_address}", response_model=ScoreResponse, tags=["scoring"])
async def score_wallet_get(
    wallet_address: str,
    include_shap: bool = True,
    _: str = Depends(_check_api_key),
) -> ScoreResponse:
    """Score an Ethereum wallet via GET request (convenient for testing)."""
    if not wallet_address.startswith("0x") or len(wallet_address) != 42:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="wallet_address must be a 42-character hex string starting with 0x",
        )
    return _build_score_response(wallet_address, include_shap)
