"""
Wallet indexer — pulls and caches raw on-chain data for a list of wallet addresses.

For each wallet we collect:
  - Normal transactions (txlist via Etherscan)
  - ERC-20 token transfers (tokentx via Etherscan)

Output: one Parquet per data type under data/raw/wallets/.

The indexer respects Etherscan's free tier (5 req/s) via a token-bucket
rate limiter and persists progress to a checkpoint file so interrupted
runs can resume.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.ethereum_client import EthereumClient

logger = logging.getLogger(__name__)

# Etherscan free tier: max 5 calls/second
_CALLS_PER_SECOND = 4.5
_CALL_INTERVAL = 1.0 / _CALLS_PER_SECOND


def _throttle(last_call: float) -> float:
    """Sleep until the rate-limit interval has passed; return new timestamp."""
    elapsed = time.monotonic() - last_call
    if elapsed < _CALL_INTERVAL:
        time.sleep(_CALL_INTERVAL - elapsed)
    return time.monotonic()


def index_wallet(
    client: EthereumClient,
    address: str,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all transaction data for a single wallet address."""
    txs = client.get_normal_transactions(address)
    token_txs = client.get_token_transfers(address)
    return {"normal_txs": txs, "token_txs": token_txs}


def index_wallets(
    client: EthereumClient,
    wallets: list[str],
    output_dir: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """Index a list of wallets and write data to Parquet files.

    Args:
        client: Authenticated Ethereum client.
        wallets: List of wallet addresses (checksummed).
        output_dir: Directory to write wallet-level Parquet files.
        checkpoint_path: Path to JSON checkpoint file tracking progress.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    done: set[str] = set()
    if checkpoint_path and checkpoint_path.exists():
        with checkpoint_path.open() as f:
            done = set(json.load(f).get("done", []))
        logger.info(f"Resuming: {len(done)} wallets already indexed")

    all_normal_txs: list[dict] = []
    all_token_txs: list[dict] = []
    last_call = 0.0

    for i, wallet in enumerate(wallets):
        if wallet in done:
            continue

        logger.info(f"[{i+1}/{len(wallets)}] Indexing {wallet}")

        try:
            last_call = _throttle(last_call)
            normal_txs = client.get_normal_transactions(wallet)
            last_call = _throttle(last_call)
            token_txs = client.get_token_transfers(wallet)

            for tx in normal_txs:
                tx["wallet"] = wallet
            for tx in token_txs:
                tx["wallet"] = wallet

            all_normal_txs.extend(normal_txs)
            all_token_txs.extend(token_txs)
            done.add(wallet)

        except Exception as exc:
            logger.error(f"Failed to index {wallet}: {exc}")
            time.sleep(2)
            continue

        # Flush every 200 wallets to avoid memory buildup
        if len(done) % 200 == 0:
            _flush(all_normal_txs, all_token_txs, output_dir, append=True)
            all_normal_txs.clear()
            all_token_txs.clear()

        # Checkpoint
        if checkpoint_path:
            with checkpoint_path.open("w") as f:
                json.dump({"done": list(done)}, f)

    # Final flush
    if all_normal_txs or all_token_txs:
        _flush(all_normal_txs, all_token_txs, output_dir, append=True)

    logger.info(f"Indexing complete. {len(done)} wallets indexed to {output_dir}")


def _flush(
    normal_txs: list[dict],
    token_txs: list[dict],
    output_dir: Path,
    append: bool = False,
) -> None:
    """Write buffered transactions to Parquet, optionally appending to existing."""
    for data, name in [(normal_txs, "normal_txs"), (token_txs, "token_txs")]:
        if not data:
            continue
        df_new = pd.DataFrame(data)
        out = output_dir / f"{name}.parquet"
        if append and out.exists():
            df_existing = pd.read_parquet(out)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(out, index=False)
        else:
            df_new.to_parquet(out, index=False)
        logger.info(f"  Flushed {len(df_new):,} rows → {out}")


def load_wallet_list(cohort_paths: list[Path]) -> list[str]:
    """Load wallet addresses from one or more Parquet cohort files."""
    dfs = []
    for p in cohort_paths:
        df = pd.read_parquet(p)
        addr_col = "borrower" if "borrower" in df.columns else "wallet"
        dfs.append(df[[addr_col]].rename(columns={addr_col: "wallet"}))

    combined = pd.concat(dfs, ignore_index=True)
    wallets = combined["wallet"].dropna().unique().tolist()
    logger.info(f"Total wallets to index: {len(wallets):,}")
    return wallets


def main() -> None:
    parser = argparse.ArgumentParser(description="Index on-chain data for wallet list")
    parser.add_argument(
        "--cohorts",
        nargs="+",
        type=Path,
        default=[
            Path("data/raw/aave_v2_liquidations.parquet"),
            Path("data/raw/non_default_cohort.parquet"),
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/wallets")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/raw/indexer_checkpoint.json"),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only index first N wallets (testing)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    client = EthereumClient.from_env()
    wallets = load_wallet_list(args.cohorts)
    if args.limit:
        wallets = wallets[: args.limit]

    index_wallets(
        client,
        wallets=wallets,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
