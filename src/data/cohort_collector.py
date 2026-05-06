"""
Non-default cohort collector — samples Aave V2 wallets that never experienced
a liquidation event during the observation window.

Strategy:
  1. Fetch all unique borrowers (Borrow events on Aave V2 LendingPool).
  2. Remove wallets already in the default cohort (liquidated).
  3. Sample up to `max_wallets` to create a balanced dataset.

The resulting cohort forms the "default = 0" class in training.
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

# keccak256("Borrow(address,address,address,uint256,uint8,uint256,uint16)")
BORROW_TOPIC = (
    "0xc6a898309e823ee50bac64e45ca8adba6690e99e7841c45d754e2a38e9019d9b"
)

# keccak256("Deposit(address,address,address,uint256,uint16)")
DEPOSIT_TOPIC = (
    "0xde6857219544bb5b7746f48ed30be6386fefc61b2f864cacf559893bf50fd951"
)

AAVE_V2_START_BLOCK = 11_500_000
BLOCK_CHUNK_SIZE = 10_000


def _parse_user_from_log(log: dict) -> str | None:
    """Extract the 'user' (borrower/depositor) from a Borrow or Deposit log.

    Borrow event indexed fields: reserve, onBehalfOf, user  (topic3 = user)
    Deposit event indexed fields: reserve, onBehalfOf, user (topic3 = user)
    """
    topics = log.get("topics", [])
    if len(topics) < 4:
        return None
    raw = topics[3].hex() if hasattr(topics[3], "hex") else topics[3]
    address = "0x" + raw[-40:]
    return Web3.to_checksum_address(address)


def collect_borrowers(
    client: EthereumClient,
    start_block: int,
    end_block: int,
    chunk_size: int = BLOCK_CHUNK_SIZE,
) -> set[str]:
    """Return all unique borrower addresses that made at least one Borrow call."""
    borrowers: set[str] = set()
    current = start_block

    while current < end_block:
        chunk_end = min(current + chunk_size, end_block)
        logger.info(f"Scanning Borrow events: blocks {current:,} → {chunk_end:,}")

        try:
            logs = client.get_logs(
                address=AAVE_V2_LENDING_POOL,
                topics=[BORROW_TOPIC],
                from_block=current,
                to_block=chunk_end,
            )
            for log in logs:
                user = _parse_user_from_log(log)
                if user:
                    borrowers.add(user)
        except Exception as exc:
            logger.error(f"Failed chunk {current}–{chunk_end}: {exc}")
            time.sleep(5)

        current = chunk_end
        time.sleep(0.2)

    logger.info(f"Total unique borrowers found: {len(borrowers):,}")
    return borrowers


def build_non_default_cohort(
    client: EthereumClient,
    liquidated_path: Path,
    output_path: Path,
    start_block: int = AAVE_V2_START_BLOCK,
    end_block: int | None = None,
    max_wallets: int = 5_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample non-default wallets and save as Parquet."""
    # Load known defaulters
    if not liquidated_path.exists():
        raise FileNotFoundError(
            f"Liquidation file not found: {liquidated_path}\n"
            "Run liquidation_collector.py first."
        )
    liquidated = pd.read_parquet(liquidated_path)
    defaulted_wallets = set(liquidated["borrower"].unique())
    logger.info(f"Known defaulted wallets: {len(defaulted_wallets):,}")

    end_block = end_block or client.get_latest_block()

    # Collect all borrowers
    all_borrowers = collect_borrowers(
        client, start_block=start_block, end_block=end_block
    )

    # Exclude defaulters
    eligible = all_borrowers - defaulted_wallets
    logger.info(f"Eligible non-default borrowers: {len(eligible):,}")

    # Sample
    import random
    random.seed(random_seed)
    sampled = random.sample(list(eligible), min(max_wallets, len(eligible)))

    df = pd.DataFrame({"wallet": sampled, "label": 0})
    df["source"] = "aave_v2_borrow"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Non-default cohort saved: {len(df):,} wallets → {output_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample non-default Aave V2 borrowers"
    )
    parser.add_argument(
        "--liquidated",
        type=Path,
        default=Path("data/raw/aave_v2_liquidations.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/non_default_cohort.parquet"),
    )
    parser.add_argument("--max-wallets", type=int, default=5_000)
    parser.add_argument("--start-block", type=int, default=AAVE_V2_START_BLOCK)
    parser.add_argument("--end-block", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    client = EthereumClient.from_env()
    build_non_default_cohort(
        client,
        liquidated_path=args.liquidated,
        output_path=args.output,
        start_block=args.start_block,
        end_block=args.end_block,
        max_wallets=args.max_wallets,
    )


if __name__ == "__main__":
    main()
