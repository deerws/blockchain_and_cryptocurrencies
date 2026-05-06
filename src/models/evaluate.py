"""
Model evaluation suite — computes industry-standard credit risk metrics and
generates all plots needed for the white paper and LinkedIn portfolio.

Metrics computed:
  - ROC-AUC (discrimination)
  - KS statistic (Kolmogorov-Smirnov, standard in credit risk)
  - Gini coefficient (= 2 * AUC - 1)
  - Brier score (calibration)
  - Lift at decile (business impact)

Plots generated (saved to reports/figures/):
  - roc_curves.png
  - calibration_plot.png
  - ks_plot.png
  - lift_chart.png
  - shap_summary.png
  - score_distribution.png
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/figures")
MODELS_DIR = Path("models")

# Brand color — matches the white paper palette
BRAND_BLUE = "#185FA5"


# ── Metric functions ───────────────────────────────────────────────────────

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between TPR and FPR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Standard credit risk discrimination metric."""
    return 2.0 * roc_auc_score(y_true, y_prob) - 1.0


def lift_at_decile(
    y_true: np.ndarray, y_prob: np.ndarray, decile: int = 1
) -> float:
    """Lift at the top N decile. Decile=1 means top 10% of predicted risk."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df = df.sort_values("p", ascending=False)
    n = len(df)
    top_n = max(1, int(n * decile / 10))
    base_rate = df["y"].mean()
    top_rate = df.head(top_n)["y"].mean()
    return top_rate / base_rate if base_rate > 0 else 0.0


# ── Plot functions ─────────────────────────────────────────────────────────

def plot_roc_curves(
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [BRAND_BLUE, "#E05C2A", "#2ABD6E"]

    for (name, prob), color in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — ChainScore Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_calibration(
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [BRAND_BLUE, "#E05C2A"]

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    for (name, prob), color in zip(probs.items(), colors):
        fraction_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        brier = brier_score_loss(y_test, prob)
        ax.plot(
            mean_pred, fraction_pos, "o-",
            color=color, lw=2, markersize=6,
            label=f"{name} (Brier = {brier:.4f})"
        )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot — ChainScore Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_ks(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """KS plot: cumulative distribution of scores for each class."""
    df = pd.DataFrame({"y": y_test, "p": y_prob}).sort_values("p")
    n = len(df)
    cum_default = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    cum_nondefault = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(n), cum_nondefault.values, color="#2ABD6E", lw=2, label="Non-default")
    ax.plot(range(n), cum_default.values, color="#E05C2A", lw=2, label="Default")

    ks = ks_statistic(y_test, y_prob)
    ax.set_title(f"KS Plot — {model_name} (KS = {ks:.3f})")
    ax.set_xlabel("Wallets sorted by predicted score")
    ax.set_ylabel("Cumulative fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_lift(
    y_test: np.ndarray,
    probs: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Lift chart at each decile."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [BRAND_BLUE, "#E05C2A"]
    deciles = list(range(1, 11))

    for (name, prob), color in zip(probs.items(), colors):
        lifts = [lift_at_decile(y_test, prob, d) for d in deciles]
        ax.plot(deciles, lifts, "o-", color=color, lw=2, markersize=7, label=name)

    ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5, label="Baseline (no model)")
    ax.set_xlabel("Decile (1 = top 10% risk)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Chart by Decile")
    ax.set_xticks(deciles)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_shap_summary(
    model,
    X_test: pd.DataFrame,
    save_path: Path,
    max_display: int = 15,
) -> None:
    """SHAP beeswarm summary plot for the LightGBM model."""
    try:
        # Extract the underlying LightGBM booster for SHAP
        inner = model
        if hasattr(model, "calibrated_classifiers_"):
            inner = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "named_steps"):
            inner = model.named_steps.get("classifier", model)

        explainer = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(X_test)

        # For binary classification, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {save_path}")
    except Exception as exc:
        logger.warning(f"SHAP plot failed: {exc}")


def plot_score_distribution(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> None:
    """Histogram of ChainScore values (0–1000) by class."""
    scores = np.round(1000 * (1 - y_prob)).astype(int)
    df = pd.DataFrame({"score": scores, "class": y_test})

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = list(range(0, 1050, 50))

    for label, color, name in [(0, "#2ABD6E", "Non-default"), (1, "#E05C2A", "Default")]:
        subset = df.loc[df["class"] == label, "score"]
        ax.hist(subset, bins=bins, alpha=0.6, color=color, label=name, density=True)

    ax.set_xlabel("ChainScore (0–1000, higher = lower risk)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Credit Class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ── Main evaluation runner ─────────────────────────────────────────────────

def run_evaluation(
    feature_matrix_path: Path = Path("data/processed/feature_matrix.parquet"),
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    with (models_dir / "logistic_regression.pkl").open("rb") as f:
        lr = pickle.load(f)
    with (models_dir / "lightgbm.pkl").open("rb") as f:
        lgb_model = pickle.load(f)
    with (models_dir / "feature_columns.json").open() as f:
        feature_cols = json.load(f)
    with (models_dir / "training_metadata.json").open() as f:
        meta = json.load(f)

    # Reload test set using the same temporal split
    df = pd.read_parquet(feature_matrix_path)
    test_mask = df["first_tx_block"] > meta.get("temporal_split_block", 17_000_000)
    if test_mask.sum() < 50:
        from sklearn.model_selection import train_test_split
        X_all = df[feature_cols]
        y_all = df["label"]
        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )
        y_test_np = y_test.values
        X_test_df = X_test
    else:
        X_test_df = df.loc[test_mask, feature_cols]
        y_test_np = df.loc[test_mask, "label"].values

    lr_prob = lr.predict_proba(X_test_df)[:, 1]
    lgb_prob = lgb_model.predict_proba(X_test_df)[:, 1]

    probs = {"Logistic Regression": lr_prob, "LightGBM": lgb_prob}

    # ── Compute all metrics ────────────────────────────────────────────────
    results = {}
    for name, prob in probs.items():
        results[name] = {
            "roc_auc": float(roc_auc_score(y_test_np, prob)),
            "ks_statistic": float(ks_statistic(y_test_np, prob)),
            "gini": float(gini_coefficient(y_test_np, prob)),
            "brier_score": float(brier_score_loss(y_test_np, prob)),
            "lift_decile_1": float(lift_at_decile(y_test_np, prob, 1)),
            "lift_decile_2": float(lift_at_decile(y_test_np, prob, 2)),
        }
        logger.info(f"\n{'='*40}\n{name}")
        for metric, val in results[name].items():
            logger.info(f"  {metric}: {val:.4f}")

    # ── Generate plots ─────────────────────────────────────────────────────
    plot_roc_curves(y_test_np, probs, reports_dir / "roc_curves.png")
    plot_calibration(y_test_np, probs, reports_dir / "calibration_plot.png")
    plot_ks(y_test_np, lgb_prob, "LightGBM", reports_dir / "ks_plot.png")
    plot_lift(y_test_np, probs, reports_dir / "lift_chart.png")
    plot_score_distribution(y_test_np, lgb_prob, reports_dir / "score_distribution.png")
    plot_shap_summary(lgb_model, X_test_df, reports_dir / "shap_summary.png")

    # Save results JSON
    with (reports_dir / "evaluation_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll plots and metrics saved to {reports_dir}")
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ChainScore models")
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/feature_matrix.parquet")
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_evaluation(
        feature_matrix_path=args.features,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
