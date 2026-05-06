"""
Liquidation collector — fetches LiquidationCall events from Aave V2.

Aave V2 LendingPool contract: 0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9

LiquidationCall event signature:
    event LiquidationCall(
        address indexed collateralAsset,
        address indexed debtAsset,
        address indexed user,           ← the liquidated borrower
        uint256 debtToCover,
        uint256 liquidatedCollateralAmount,
        address liquidator,
        bool receiveAToken
    )

Topic0 (event hash):
    0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286

This script outputs a Parquet file with one row per liquidation event, including
the borrower's wallet address, timestamp, block number, and the assets involved.
These borrowers form the "default = 1" class in our training set.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from web3 import Web3

from src.data.ethereum_client import EthereumClient

logger = logging.getLogger(__name__)

# ── Aave V2 constants ──────────────────────────────────────────────────────
AAVE_V2_LENDING_POOL = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"
LIQUIDATION_CALL_TOPIC = (
    "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286"
)

# Aave V2 was deployed at block 11,362,579 (Dec 2020). We start a bit later
# to avoid the initial bootstrap period where data is sparse.
AAVE_V2_START_BLOCK = 11_500_000

# Block range chunk size — Alchemy free tier limits queries to ~10k blocks
# per call to avoid timeouts.
BLOCK_CHUNK_SIZE = 10_000


def parse_liquidation_log(log: dict, web3: Web3) -> dict:
    """Decode a LiquidationCall log into a structured dict."""
    # Indexed topics are: collateralAsset, debtAsset, user (borrower)
    collateral_asset = "0x" + log["topics"][1].hex()[-40:]
    debt_asset = "0x" + log["topics"][2].hex()[-40:]
    borrower = "0x" + log["topics"][3].hex()[-40:]

    # Non-indexed data: debtToCover, liquidatedCollateralAmount, liquidator, receiveAToken
    # We only need the essential fields for now; full decoding can come later.
    return {
        "block_number": log["blockNumber"],
        "tx_hash": log["transactionHash"].hex(),
        "log_index": log["logIndex"],
        "borrower": Web3.to_checksum_address(borrower),
        "collateral_asset": Web3.to_checksum_address(collateral_asset),
        "debt_asset": Web3.to_checksum_address(debt_asset),
    }


def collect_liquidations(
    client: EthereumClient,
    start_block: int,
    end_block: int,
    output_path: Path,
    chunk_size: int = BLOCK_CHUNK_SIZE,
) -> pd.DataFrame:
    """Iterate block ranges and collect all Aave V2 liquidations."""
    all_liquidations = []
    current = start_block

    while current < end_block:
        chunk_end = min(current + chunk_size, end_block)
        logger.info(f"Fetching blocks {current:,} → {chunk_end:,}")

        try:
            logs = client.get_logs(
                address=AAVE_V2_LENDING_POOL,
                topics=[LIQUIDATION_CALL_TOPIC],
                from_block=current,
                to_block=chunk_end,
            )
        except Exception as e:
            logger.error(f"Failed chunk {current}–{chunk_end}: {e}")
            time.sleep(5)
            current = chunk_end
            continue

        for log in logs:
            all_liquidations.append(parse_liquidation_log(log, client.web3))

        logger.info(f"  Found {len(logs)} liquidations in this chunk")
        current = chunk_end

        # Light rate limiting to be polite to Alchemy
        time.sleep(0.2)

    df = pd.DataFrame(all_liquidations)
    if df.empty:
        logger.warning("No liquidations found.")
        return df

    df = df.drop_duplicates(subset=["tx_hash", "log_index"])
    logger.info(f"Total unique liquidations: {len(df):,}")
    logger.info(f"Unique borrowers: {df['borrower'].nunique():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Aave V2 liquidation events")
    parser.add_argument(
        "--start-block",
        type=int,
        default=AAVE_V2_START_BLOCK,
        help="Block to start scanning from",
    )
    parser.add_argument(
        "--end-block",
        type=int,
        default=None,
        help="Block to stop at (default: latest)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N blocks (useful for testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/aave_v2_liquidations.parquet"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    client = EthereumClient.from_env()
    end_block = args.end_block or client.get_latest_block()

    if args.limit:
        end_block = min(end_block, args.start_block + args.limit)

    logger.info(f"Collecting liquidations from block {args.start_block:,} to {end_block:,}")

    collect_liquidations(
        client=client,
        start_block=args.start_block,
        end_block=end_block,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
