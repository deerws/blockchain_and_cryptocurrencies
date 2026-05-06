"""
Feature builder — transforms raw on-chain transaction data into a structured
feature matrix for credit risk modeling.

Implements 45 features across 5 families:
  1. Transaction Volume  (9 features)
  2. Counterparty Graph  (7 features)
  3. Protocol Diversity  (7 features)
  4. Collateral Behavior (8 features)
  5. Temporal Consistency (14 features)

All monetary values are normalized to ETH (not Wei). Timestamps are in UTC.

Usage:
    python -m src.features.builder
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

from src.features.protocol_registry import detect_protocols

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ──────────────────────────────────────────────────────────────
WEI_PER_ETH = 1e18
SECONDS_PER_DAY = 86_400
RECENT_WINDOW_DAYS = 90

# Aave V2 addresses — used to count Aave-specific interactions
AAVE_V2_ADDRESSES = {
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",
}


# ── Helper utilities ───────────────────────────────────────────────────────

def _to_eth(wei_series: pd.Series) -> pd.Series:
    return pd.to_numeric(wei_series, errors="coerce").fillna(0) / WEI_PER_ETH


def _to_ts(unix_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(unix_series, errors="coerce"), unit="s", utc=True
    )


# ── Feature family builders ────────────────────────────────────────────────

def _volume_features(txs: pd.DataFrame, wallet: str) -> dict:
    """9 features — raw transaction volume and value statistics."""
    if txs.empty:
        return {f: 0.0 for f in [
            "tx_count", "eth_sent_total", "eth_received_total", "net_eth_flow",
            "avg_tx_value_eth", "std_tx_value_eth", "max_tx_value_eth",
            "failed_tx_count", "failed_tx_ratio",
        ]}

    values = _to_eth(txs["value"])
    is_sender = txs["from"].str.lower() == wallet.lower()
    is_receiver = txs["to"].str.lower() == wallet.lower()
    is_failed = txs.get("isError", pd.Series(["0"] * len(txs))) == "1"

    tx_count = len(txs)
    failed_count = int(is_failed.sum())

    return {
        "tx_count": tx_count,
        "eth_sent_total": float(values[is_sender].sum()),
        "eth_received_total": float(values[is_receiver].sum()),
        "net_eth_flow": float(values[is_receiver].sum() - values[is_sender].sum()),
        "avg_tx_value_eth": float(values.mean()),
        "std_tx_value_eth": float(values.std(ddof=0)),
        "max_tx_value_eth": float(values.max()),
        "failed_tx_count": failed_count,
        "failed_tx_ratio": failed_count / tx_count if tx_count else 0.0,
    }


def _graph_features(txs: pd.DataFrame, wallet: str) -> dict:
    """7 features — counterparty diversity and concentration."""
    if txs.empty:
        return {f: 0.0 for f in [
            "unique_senders", "unique_receivers", "unique_counterparties",
            "top1_concentration", "contract_interactions",
            "contract_ratio", "self_tx_ratio",
        ]}

    w = wallet.lower()
    senders = txs["from"].str.lower()
    receivers = txs["to"].str.lower()

    unique_senders = int(senders[senders != w].nunique())
    unique_receivers = int(receivers[receivers != w].nunique())
    all_cp = pd.concat([senders[senders != w], receivers[receivers != w]])
    unique_cp = int(all_cp.nunique())

    # Top-1 concentration: fraction of txs to/from single most common counterparty
    cp_counts = all_cp.value_counts()
    top1_conc = float(cp_counts.iloc[0] / len(txs)) if not cp_counts.empty else 0.0

    # Contract interactions: txs where input data is not empty (non-plain ETH transfer)
    has_input = txs.get("input", pd.Series(["0x"] * len(txs)))
    contract_interactions = int((has_input != "0x").sum())
    contract_ratio = contract_interactions / len(txs) if len(txs) else 0.0

    self_tx = int((senders == w) & (receivers == w)).sum() if len(txs) else 0
    self_ratio = self_tx / len(txs) if len(txs) else 0.0

    return {
        "unique_senders": unique_senders,
        "unique_receivers": unique_receivers,
        "unique_counterparties": unique_cp,
        "top1_concentration": top1_conc,
        "contract_interactions": contract_interactions,
        "contract_ratio": contract_ratio,
        "self_tx_ratio": self_ratio,
    }


def _protocol_features(txs: pd.DataFrame, token_txs: pd.DataFrame) -> dict:
    """7 features — DeFi protocol breadth and diversity."""
    all_addresses: list[str] = []

    if not txs.empty:
        all_addresses += txs["to"].dropna().str.lower().tolist()
    if not token_txs.empty:
        all_addresses += token_txs["contractAddress"].dropna().str.lower().tolist()
        all_addresses += token_txs["to"].dropna().str.lower().tolist()

    protocols = detect_protocols(all_addresses)

    # Protocol-specific counts
    aave_txs = sum(1 for a in all_addresses if a in AAVE_V2_ADDRESSES)
    is_aave_user = int("aave_v2" in protocols or "aave_v1" in protocols)
    is_compound_user = int("compound_v2" in protocols)
    is_uniswap_user = int("uniswap_v2" in protocols or "uniswap_v3" in protocols)

    n_protocols = len(protocols)
    # Diversity index: Shannon entropy over protocol interaction counts
    if not txs.empty and n_protocols > 1:
        all_addr_series = pd.Series(all_addresses)
        protocol_hits = []
        from src.features.protocol_registry import ADDRESS_TO_PROTOCOL
        for a in all_addr_series:
            if a in ADDRESS_TO_PROTOCOL:
                protocol_hits.append(ADDRESS_TO_PROTOCOL[a])
        if protocol_hits:
            counts = pd.Series(protocol_hits).value_counts(normalize=True)
            diversity_index = float(entropy(counts.values, base=2))
        else:
            diversity_index = 0.0
    else:
        diversity_index = 0.0

    unique_tokens = 0
    if not token_txs.empty:
        unique_tokens = int(token_txs["contractAddress"].nunique())

    return {
        "protocols_used_count": n_protocols,
        "is_aave_user": is_aave_user,
        "is_compound_user": is_compound_user,
        "is_uniswap_user": is_uniswap_user,
        "aave_interaction_count": aave_txs,
        "protocol_diversity_index": diversity_index,
        "unique_erc20_tokens": unique_tokens,
    }


def _collateral_features(txs: pd.DataFrame, wallet: str) -> dict:
    """8 features — approximated collateral and repayment behavior from gas patterns
    and known Aave function selectors.

    Since we don't decode ABI in this pipeline, we use function selectors (first 4 bytes
    of `input`) to identify Aave/Compound calls. This is a known-limited proxy.
    """
    AAVE_DEPOSIT_SEL = "0xe8eda9df"   # deposit(address,uint256,address,uint16)
    AAVE_BORROW_SEL  = "0xa415bcad"   # borrow(address,uint256,uint256,uint16,address)
    AAVE_REPAY_SEL   = "0x573ade81"   # repay(address,uint256,uint256,address)
    AAVE_WITHDRAW_SEL = "0x69328dec"  # withdraw(address,uint256,address)

    if txs.empty:
        return {f: 0.0 for f in [
            "aave_deposit_count", "aave_borrow_count", "aave_repay_count",
            "aave_withdraw_count", "repay_to_borrow_ratio",
            "avg_gas_price_gwei", "gas_price_percentile_75",
            "high_gas_tx_ratio",
        ]}

    inp = txs.get("input", pd.Series(["0x"] * len(txs))).fillna("0x")
    selectors = inp.str[:10].str.lower()

    deposit_count  = int((selectors == AAVE_DEPOSIT_SEL).sum())
    borrow_count   = int((selectors == AAVE_BORROW_SEL).sum())
    repay_count    = int((selectors == AAVE_REPAY_SEL).sum())
    withdraw_count = int((selectors == AAVE_WITHDRAW_SEL).sum())

    repay_ratio = repay_count / borrow_count if borrow_count else 0.0

    gas_price_gwei = pd.to_numeric(txs.get("gasPrice", 0), errors="coerce") / 1e9
    avg_gas = float(gas_price_gwei.mean()) if not gas_price_gwei.empty else 0.0
    p75_gas = float(gas_price_gwei.quantile(0.75)) if not gas_price_gwei.empty else 0.0
    high_gas_ratio = float((gas_price_gwei > gas_price_gwei.quantile(0.9)).mean()) if not gas_price_gwei.empty else 0.0

    return {
        "aave_deposit_count": deposit_count,
        "aave_borrow_count": borrow_count,
        "aave_repay_count": repay_count,
        "aave_withdraw_count": withdraw_count,
        "repay_to_borrow_ratio": repay_ratio,
        "avg_gas_price_gwei": avg_gas,
        "gas_price_percentile_75": p75_gas,
        "high_gas_tx_ratio": high_gas_ratio,
    }


def _temporal_features(txs: pd.DataFrame) -> dict:
    """14 features — temporal consistency and activity regularity."""
    empty = {f: 0.0 for f in [
        "wallet_age_days", "active_days_count", "activity_span_days",
        "avg_days_between_txs", "std_days_between_txs",
        "tx_burst_coefficient", "recent_tx_ratio",
        "active_months_count", "activity_regularity",
        "dormancy_periods_30d", "weekday_ratio",
        "first_tx_block", "last_tx_block", "block_span",
    ]}

    if txs.empty:
        return empty

    ts = _to_ts(txs["timeStamp"])
    ts_sorted = ts.dropna().sort_values()

    if ts_sorted.empty:
        return empty

    first = ts_sorted.iloc[0]
    last = ts_sorted.iloc[-1]

    wallet_age_days = (last - first).total_seconds() / SECONDS_PER_DAY
    activity_span = wallet_age_days  # same for now

    dates = ts_sorted.dt.date
    active_days = int(dates.nunique())

    # Inter-transaction gaps
    if len(ts_sorted) > 1:
        gaps_days = ts_sorted.diff().dt.total_seconds().dropna() / SECONDS_PER_DAY
        avg_gap = float(gaps_days.mean())
        std_gap = float(gaps_days.std(ddof=0))
        dormancy = int((gaps_days > 30).sum())
    else:
        avg_gap, std_gap, dormancy = 0.0, 0.0, 0

    # Burst coefficient: max daily tx count / avg daily tx count
    daily_counts = ts_sorted.dt.date.value_counts()
    if len(daily_counts) > 0:
        burst = float(daily_counts.max() / daily_counts.mean())
    else:
        burst = 0.0

    # Recent activity ratio (last 90 days from observation window = last tx date)
    cutoff = last - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    recent_ratio = float((ts_sorted >= cutoff).sum() / len(ts_sorted))

    # Monthly activity
    active_months = int(ts_sorted.dt.to_period("M").nunique())
    if active_months > 1 and wallet_age_days > 0:
        months_span = wallet_age_days / 30.0
        monthly_counts = ts_sorted.dt.to_period("M").value_counts()
        activity_regularity = float(monthly_counts.std() / monthly_counts.mean()) if monthly_counts.mean() > 0 else 0.0
    else:
        activity_regularity = 0.0

    # Weekday ratio
    weekday_ratio = float((ts_sorted.dt.dayofweek < 5).sum() / len(ts_sorted))

    block_nums = pd.to_numeric(txs.get("blockNumber", 0), errors="coerce").dropna()
    first_block = int(block_nums.min()) if not block_nums.empty else 0
    last_block = int(block_nums.max()) if not block_nums.empty else 0
    block_span = last_block - first_block

    return {
        "wallet_age_days": wallet_age_days,
        "active_days_count": active_days,
        "activity_span_days": activity_span,
        "avg_days_between_txs": avg_gap,
        "std_days_between_txs": std_gap,
        "tx_burst_coefficient": burst,
        "recent_tx_ratio": recent_ratio,
        "active_months_count": active_months,
        "activity_regularity": activity_regularity,
        "dormancy_periods_30d": dormancy,
        "weekday_ratio": weekday_ratio,
        "first_tx_block": first_block,
        "last_tx_block": last_block,
        "block_span": block_span,
    }


# ── Main feature builder ───────────────────────────────────────────────────

def build_features_for_wallet(
    wallet: str,
    txs: pd.DataFrame,
    token_txs: pd.DataFrame,
) -> dict:
    """Compute all 45 features for a single wallet."""
    features = {"wallet": wallet}
    features.update(_volume_features(txs, wallet))
    features.update(_graph_features(txs, wallet))
    features.update(_protocol_features(txs, token_txs))
    features.update(_collateral_features(txs, wallet))
    features.update(_temporal_features(txs))
    return features


def build_feature_matrix(
    normal_txs_path: Path,
    token_txs_path: Path,
    labeled_default_path: Path,
    labeled_nondefault_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build the full labeled feature matrix from raw data files."""
    logger.info("Loading raw transaction data...")
    normal_txs = pd.read_parquet(normal_txs_path)
    token_txs = pd.read_parquet(token_txs_path)

    logger.info("Loading labels...")
    defaults = pd.read_parquet(labeled_default_path)[["borrower"]].rename(
        columns={"borrower": "wallet"}
    )
    defaults["label"] = 1

    non_defaults = pd.read_parquet(labeled_nondefault_path)[["wallet"]]
    non_defaults["label"] = 0

    all_wallets = pd.concat([defaults, non_defaults], ignore_index=True)
    all_wallets = all_wallets.drop_duplicates(subset="wallet")
    logger.info(f"Building features for {len(all_wallets):,} wallets...")

    rows = []
    for _, row in all_wallets.iterrows():
        wallet = row["wallet"]
        w_txs = normal_txs[normal_txs["wallet"] == wallet]
        w_token_txs = token_txs[token_txs["wallet"] == wallet]
        feats = build_features_for_wallet(wallet, w_txs, w_token_txs)
        feats["label"] = row["label"]
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Final cleanup
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Feature matrix saved: {df.shape} → {output_path}")

    return df


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build feature matrix from raw data")
    parser.add_argument(
        "--normal-txs", type=Path, default=Path("data/raw/wallets/normal_txs.parquet")
    )
    parser.add_argument(
        "--token-txs", type=Path, default=Path("data/raw/wallets/token_txs.parquet")
    )
    parser.add_argument(
        "--defaults",
        type=Path,
        default=Path("data/raw/aave_v2_liquidations.parquet"),
    )
    parser.add_argument(
        "--non-defaults",
        type=Path,
        default=Path("data/raw/non_default_cohort.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/feature_matrix.parquet"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    build_feature_matrix(
        normal_txs_path=args.normal_txs,
        token_txs_path=args.token_txs,
        labeled_default_path=args.defaults,
        labeled_nondefault_path=args.non_defaults,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
