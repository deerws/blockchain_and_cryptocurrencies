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

Uses Etherscan Logs API (not eth_getLogs) — supports up to 100k blocks per
request and is not subject to Alchemy's 10-block free-tier restriction.

Output: Parquet with one row per liquidation event including borrower address,
block number, timestamp, and asset pair. Borrowers form the "default = 1" class.
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

# Aave V2 deployed at block 11,362,579 (Dec 2020). Start a bit later to
# skip the sparse bootstrap period.
AAVE_V2_START_BLOCK = 11_500_000

# Etherscan Logs API: 100k blocks per chunk, 1000 records per page
BLOCK_CHUNK_SIZE = 100_000
PAGE_SIZE = 1_000


def _parse_log(log: dict) -> dict | None:
    """Decode a LiquidationCall log into a structured dict."""
    topics = log.get("topics", [])
    if len(topics) < 4:
        return None

    def extract_address(topic: str) -> str:
        return Web3.to_checksum_address("0x" + topic[-40:])

    return {
        "block_number": int(log["blockNumber"], 16),
        "timestamp": int(log["timeStamp"], 16),
        "tx_hash": log["transactionHash"],
        "log_index": int(log["logIndex"], 16),
        "borrower": extract_address(topics[3]),
        "collateral_asset": extract_address(topics[1]),
        "debt_asset": extract_address(topics[2]),
    }


def collect_liquidations(
    client: EthereumClient,
    start_block: int,
    end_block: int,
    output_path: Path,
    chunk_size: int = BLOCK_CHUNK_SIZE,
) -> pd.DataFrame:
    """Collect all Aave V2 LiquidationCall events via Etherscan Logs API."""
    all_liquidations: list[dict] = []
    current = start_block

    while current < end_block:
        chunk_end = min(current + chunk_size, end_block)
        logger.info(f"Fetching blocks {current:,} → {chunk_end:,}")

        page = 1
        while True:
            try:
                logs = client.get_event_logs(
                    address=AAVE_V2_LENDING_POOL,
                    topic0=LIQUIDATION_CALL_TOPIC,
                    from_block=current,
                    to_block=chunk_end,
                    page=page,
                    offset=PAGE_SIZE,
                )
            except Exception as exc:
                logger.error(f"  Page {page} failed: {exc}")
                time.sleep(3)
                break

            parsed = [r for r in (_parse_log(l) for l in logs) if r]
            all_liquidations.extend(parsed)
            logger.info(f"  Page {page}: {len(logs)} logs, {len(parsed)} parsed")

            if len(logs) < PAGE_SIZE:
                break
            page += 1
            time.sleep(0.25)  # Etherscan free: 5 req/s

        current = chunk_end
        time.sleep(0.25)

    if not all_liquidations:
        logger.warning("No liquidations found in range.")
        return pd.DataFrame()

    df = pd.DataFrame(all_liquidations)
    df = df.drop_duplicates(subset=["tx_hash", "log_index"])
    df = df.sort_values("block_number").reset_index(drop=True)

    logger.info(f"Total unique liquidations: {len(df):,}")
    logger.info(f"Unique borrowers (default=1): {df['borrower'].nunique():,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Aave V2 liquidation events")
    parser.add_argument("--start-block", type=int, default=AAVE_V2_START_BLOCK)
    parser.add_argument("--end-block", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N blocks from start (testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/aave_v2_liquidations.parquet"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    client = EthereumClient.from_env()
    end_block = args.end_block or client.get_latest_block()
    if args.limit:
        end_block = min(end_block, args.start_block + args.limit)

    logger.info(
        f"Collecting liquidations: blocks {args.start_block:,} → {end_block:,}"
        f" ({(end_block - args.start_block):,} blocks)"
    )
    collect_liquidations(
        client=client,
        start_block=args.start_block,
        end_block=end_block,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
