"""
Ethereum client — unified wrapper around Alchemy RPC and Etherscan API.

Centralizes rate limiting, retries, and error handling so the rest of the
codebase doesn't have to think about these concerns.
"""
from __future__ import annotations

import os
import logging
from typing import Any
from dataclasses import dataclass

import requests
from web3 import Web3
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
DEFAULT_TIMEOUT = 30  # seconds


@dataclass
class EthereumClient:
    """Unified Ethereum data client."""

    alchemy_rpc_url: str
    etherscan_api_key: str
    _web3: Web3 | None = None

    @classmethod
    def from_env(cls) -> "EthereumClient":
        """Build client from environment variables."""
        alchemy_key = os.getenv("ALCHEMY_API_KEY")
        etherscan_key = os.getenv("ETHERSCAN_API_KEY")

        if not alchemy_key or not etherscan_key:
            raise ValueError(
                "Missing API keys. Copy .env.example to .env and fill in your keys."
            )

        rpc_url = f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_key}"
        return cls(alchemy_rpc_url=rpc_url, etherscan_api_key=etherscan_key)

    @property
    def web3(self) -> Web3:
        """Lazy-loaded Web3 instance."""
        if self._web3 is None:
            self._web3 = Web3(Web3.HTTPProvider(self.alchemy_rpc_url))
            if not self._web3.is_connected():
                raise ConnectionError("Failed to connect to Alchemy RPC")
        return self._web3

    # ── Etherscan API ──────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _etherscan_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make a request to Etherscan with retry logic."""
        params["apikey"] = self.etherscan_api_key
        response = requests.get(
            ETHERSCAN_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        # Etherscan returns status="0" for empty results AND for errors
        # We let the caller decide how to handle empty results
        if data.get("status") == "0" and data.get("message") not in (
            "No transactions found",
            "No records found",
        ):
            logger.warning(f"Etherscan API issue: {data.get('message')}")

        return data

    def get_normal_transactions(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> list[dict[str, Any]]:
        """Get all normal transactions for a wallet address."""
        data = self._etherscan_call(
            {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "endblock": end_block,
                "sort": "asc",
            }
        )
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def get_internal_transactions(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> list[dict[str, Any]]:
        """Get internal transactions (contract-initiated) for a wallet."""
        data = self._etherscan_call(
            {
                "module": "account",
                "action": "txlistinternal",
                "address": address,
                "startblock": start_block,
                "endblock": end_block,
                "sort": "asc",
            }
        )
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def get_token_transfers(
        self, address: str, start_block: int = 0, end_block: int = 99999999
    ) -> list[dict[str, Any]]:
        """Get ERC-20 token transfers for a wallet."""
        data = self._etherscan_call(
            {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": start_block,
                "endblock": end_block,
                "sort": "asc",
            }
        )
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    # ── Web3 RPC ────────────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def get_logs(
        self,
        address: str,
        topics: list[str] | None = None,
        from_block: int = 0,
        to_block: int | str = "latest",
    ) -> list[dict[str, Any]]:
        """Fetch event logs from a contract address."""
        filter_params = {
            "address": Web3.to_checksum_address(address),
            "fromBlock": from_block,
            "toBlock": to_block,
        }
        if topics:
            filter_params["topics"] = topics

        logs = self.web3.eth.get_logs(filter_params)
        return [dict(log) for log in logs]

    def get_latest_block(self) -> int:
        """Return current latest block number."""
        return self.web3.eth.block_number


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    client = EthereumClient.from_env()
    print(f"Connected to Ethereum. Latest block: {client.get_latest_block()}")

    # Test with Vitalik's wallet
    vitalik = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    txs = client.get_normal_transactions(vitalik)
    print(f"Vitalik has {len(txs)} normal transactions")
