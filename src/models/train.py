"""
Model training pipeline — trains and calibrates two credit risk models:
  1. Logistic Regression   (interpretable baseline — scorecard)
  2. LightGBM              (gradient boosting — production model)

Both models are calibrated with Platt scaling (CalibratedClassifierCV) to
produce reliable probability estimates, which is critical for correct PD
computation.

Train/test split strategy: temporal — wallets with first interaction before
the cutoff block go into train, those after into test. This avoids data
leakage from future events informing past predictions.

Outputs:
    models/logistic_regression.pkl
    models/lightgbm.pkl
    models/feature_columns.json
    models/training_metadata.json
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

# Features excluded from the model (identifiers and leakage risks)
NON_FEATURE_COLS = {"wallet", "label", "first_tx_block", "last_tx_block"}

# Temporal split: wallets first active before this block → train;
# wallets first active after → test. Block 17,000,000 ≈ April 2023.
TEMPORAL_SPLIT_BLOCK = 17_000_000


def load_and_split(feature_matrix_path: Path) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]
]:
    """Load feature matrix and apply temporal train/test split."""
    df = pd.read_parquet(feature_matrix_path)
    logger.info(f"Loaded feature matrix: {df.shape} (label balance: {df['label'].mean():.2%} default)")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Temporal split on first_tx_block
    train_mask = df["first_tx_block"] <= TEMPORAL_SPLIT_BLOCK
    test_mask = ~train_mask

    # Fallback: if temporal split is very unbalanced, do 80/20 on label-stratified sample
    if train_mask.sum() < 100 or test_mask.sum() < 50:
        logger.warning(
            "Temporal split produced insufficient samples. Falling back to 80/20 stratified split."
        )
        from sklearn.model_selection import train_test_split
        X = df[feature_cols]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test, feature_cols

    X_train = df.loc[train_mask, feature_cols]
    X_test  = df.loc[test_mask,  feature_cols]
    y_train = df.loc[train_mask, "label"]
    y_test  = df.loc[test_mask,  "label"]

    logger.info(
        f"Train: {len(X_train):,} wallets | Test: {len(X_test):,} wallets"
    )
    logger.info(
        f"Train default rate: {y_train.mean():.2%} | Test default rate: {y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test, feature_cols


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Train a calibrated Logistic Regression pipeline."""
    logger.info("Training Logistic Regression...")

    base = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", calibrated),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> CalibratedClassifierCV:
    """Train a calibrated LightGBM model."""
    logger.info("Training LightGBM...")

    # Handle class imbalance: scale_pos_weight = ratio of negatives to positives
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    base_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        base_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        base_lgb.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(base_lgb, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved: {path}")


def train_and_save(
    feature_matrix_path: Path = Path("data/processed/feature_matrix.parquet"),
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Full training pipeline. Returns evaluation metrics on test set."""
    X_train, X_test, y_train, y_test, feature_cols = load_and_split(
        feature_matrix_path
    )

    # ── Train models ───────────────────────────────────────────────────────
    lr = train_logistic_regression(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)

    # ── Quick eval on test set ─────────────────────────────────────────────
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
    logger.info(f"Logistic Regression ROC-AUC (test): {lr_auc:.4f}")
    logger.info(f"LightGBM ROC-AUC (test): {lgb_auc:.4f}")

    # ── Persist ────────────────────────────────────────────────────────────
    save_model(lr, models_dir / "logistic_regression.pkl")
    save_model(lgb_model, models_dir / "lightgbm.pkl")

    meta = {
        "feature_columns": feature_cols,
        "temporal_split_block": TEMPORAL_SPLIT_BLOCK,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_default_rate": float(y_train.mean()),
        "test_default_rate": float(y_test.mean()),
        "lr_roc_auc": lr_auc,
        "lgb_roc_auc": lgb_auc,
    }

    with (models_dir / "feature_columns.json").open("w") as f:
        json.dump(feature_cols, f, indent=2)
    with (models_dir / "training_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete.")
    return meta


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train ChainScore credit risk models")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/feature_matrix.parquet"),
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    train_and_save(
        feature_matrix_path=args.features,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
