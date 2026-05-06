"""
Real-time inference — score a single Ethereum wallet address on demand.

This module is used by the FastAPI service (src/api/main.py) and can also
be called from the CLI for quick checks.

Pipeline:
  1. Fetch raw on-chain data for the wallet via EthereumClient.
  2. Build features using the same logic as training (FeatureBuilder).
  3. Run the LightGBM model (or Logistic Regression as fallback).
  4. Return a ScoreResult with PD, score, tier, and SHAP top factors.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ethereum_client import EthereumClient
from src.features.builder import build_features_for_wallet
from src.models.scorecard import ScoreResult

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class ChainScorePredictor:
    """Loads trained models and scores wallets on demand."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._lgb_model = None
        self._lr_model = None
        self._feature_cols: list[str] | None = None

    def _load(self) -> None:
        """Lazy-load models on first call."""
        if self._lgb_model is not None:
            return

        lgb_path = self.models_dir / "lightgbm.pkl"
        lr_path = self.models_dir / "logistic_regression.pkl"
        cols_path = self.models_dir / "feature_columns.json"

        if not lgb_path.exists():
            raise FileNotFoundError(
                f"Model not found at {lgb_path}. Run `python -m src.models.train` first."
            )

        with lgb_path.open("rb") as f:
            self._lgb_model = pickle.load(f)
        with lr_path.open("rb") as f:
            self._lr_model = pickle.load(f)
        with cols_path.open() as f:
            self._feature_cols = json.load(f)

        logger.info("Models loaded successfully.")

    def _get_shap_factors(
        self, model, X: pd.DataFrame, top_n: int = 5
    ) -> list[dict]:
        """Extract top SHAP factors for model explainability."""
        try:
            import shap
            inner = model
            if hasattr(model, "calibrated_classifiers_"):
                inner = model.calibrated_classifiers_[0].estimator

            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_row = shap_values[0]
            feature_names = X.columns.tolist()
            pairs = sorted(
                zip(feature_names, shap_row),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]

            return [
                {
                    "feature": name,
                    "shap_value": float(val),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                }
                for name, val in pairs
            ]
        except Exception as exc:
            logger.debug(f"SHAP computation failed: {exc}")
            return []

    def score_wallet(
        self,
        wallet_address: str,
        client: EthereumClient,
        use_shap: bool = True,
    ) -> ScoreResult:
        """Fetch on-chain data and compute a ChainScore for the wallet."""
        self._load()

        logger.info(f"Scoring wallet: {wallet_address}")

        # Fetch raw data
        txs_raw = client.get_normal_transactions(wallet_address)
        token_txs_raw = client.get_token_transfers(wallet_address)

        txs = pd.DataFrame(txs_raw) if txs_raw else pd.DataFrame()
        token_txs = pd.DataFrame(token_txs_raw) if token_txs_raw else pd.DataFrame()

        if txs.empty:
            logger.warning(f"No transactions found for {wallet_address}. Using zero-feature vector.")

        # Add wallet column for builder compatibility
        if not txs.empty:
            txs["wallet"] = wallet_address
        if not token_txs.empty:
            token_txs["wallet"] = wallet_address

        # Build features
        feat_dict = build_features_for_wallet(wallet_address, txs, token_txs)
        X = pd.DataFrame([feat_dict])[self._feature_cols].fillna(0)

        # Predict PD
        pd_prob = float(self._lgb_model.predict_proba(X)[0, 1])

        # SHAP factors
        factors = self._get_shap_factors(self._lgb_model, X) if use_shap else []

        return ScoreResult.from_pd(
            wallet_address=wallet_address,
            pd=pd_prob,
            top_factors=factors,
            model_version="lgbm_v1",
        )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Score a single Ethereum wallet")
    parser.add_argument("wallet", help="Ethereum wallet address (0x...)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    client = EthereumClient.from_env()
    predictor = ChainScorePredictor(models_dir=args.models_dir)
    result = predictor.score_wallet(args.wallet, client)

    import json
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
