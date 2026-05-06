"""
Known DeFi protocol contract addresses on Ethereum mainnet.

Used by the feature builder to detect protocol-level interactions from
raw transaction data (to/from addresses in normal_txs and token_txs).
"""
from __future__ import annotations

# Maps protocol name → set of known contract addresses (lowercase)
PROTOCOL_ADDRESSES: dict[str, set[str]] = {
    "aave_v1": {
        "0x398ec7346dcd622edc5ae82352f02be94c62d119",  # LendingPool
        "0x3dfd23a6c5e8bbcfc9581d2e864a68feb6a076d3",  # LendingPoolCore
    },
    "aave_v2": {
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # LendingPool
        "0x057835ad21a177dbdd3090bb1cae03eacf78fc6d",  # IncentivesController
        "0x311bb771e4f8952e6da169b425e7e92d6ac45756",  # ProtocolDataProvider
    },
    "aave_v3": {
        "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",  # Pool
        "0x64b761d848206f447fe2dd461b0c635ec39ebb27",  # PoolConfigurator
    },
    "compound_v2": {
        "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",  # Comptroller
        "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5",  # cETH
        "0x5d3a536e4d6dbd6114cc1ead35777bab948e3643",  # cDAI
        "0x39aa39c021dfbae8fac545936693ac917d5e7563",  # cUSDC
    },
    "uniswap_v2": {
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Router02
        "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f",  # Factory
    },
    "uniswap_v3": {
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # SwapRouter
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # SwapRouter02
        "0x1f98431c8ad98523631ae4a59f267346ea31f984",  # Factory
        "0xc36442b4a4522e871399cd717abdd847ab11fe88",  # NonfungiblePositionManager
    },
    "curve": {
        "0xd51a44d3fae010294c616388b506acda1bfaae46",  # TriCrypto2
        "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7",  # 3pool
        "0xdc24316b9ae028f1497c275eb9192a3ea0f67022",  # stETH pool
        "0xa5407eae9ba41422680e2e00537571bcc53efbfd",  # sUSD pool
    },
    "makerdao": {
        "0x35d1b3f3d7966a1dfe207aa4514c12a259a0492b",  # Vat
        "0x9759a6ac90977b93b58547b4a71c78317f391a28",  # DaiJoin
        "0xa26e80e7dea86279c6d778d702cc413e6cffa777",  # MCD_JOIN_ETH_A
        "0x83f20f44975d03b1b09e64809b757c47f942beea",  # sDAI
    },
    "lido": {
        "0xae7ab96520de3a18e5e111b5eaab095312d7fe84",  # stETH
        "0x889edc2edab5f40e902b864ad4d7ade8e412f9b1",  # WithdrawalQueue
    },
    "convex": {
        "0xf403c135812408bfbe8713b5a23a04b3d48aae31",  # Booster
        "0xcf50b810e57ac33b91dcf525c6ddd9881b139332",  # cvxCRV Staking
    },
    "yearn": {
        "0x9d409a0a012cfba9b15f6d4b36ac57a46966ab9a",  # YearnRegistry
    },
    "1inch": {
        "0x1111111254eeb25477b68fb85ed929f73a960582",  # AggregationRouterV5
        "0x1111111254fb6c44bac0bed2854e76f90643097d",  # AggregationRouterV4
    },
    "balancer_v2": {
        "0xba12222222228d8ba445958a75a0704d566bf2c8",  # Vault
    },
    "gnosis_safe": {
        "0xa6b71e26c5e0845f74c812102ca7114b6a896ab2",  # GnosisSafeFactory
        "0xd9db270c1b5e3bd161e8c8503c55ceabee709552",  # GnosisSafe singleton
    },
}

# Flat lookup: address → protocol name
ADDRESS_TO_PROTOCOL: dict[str, str] = {
    addr: protocol
    for protocol, addresses in PROTOCOL_ADDRESSES.items()
    for addr in addresses
}

ALL_PROTOCOL_ADDRESSES: frozenset[str] = frozenset(ADDRESS_TO_PROTOCOL.keys())


def detect_protocols(addresses: list[str]) -> set[str]:
    """Return the set of protocol names interacted with from a list of addresses."""
    return {
        ADDRESS_TO_PROTOCOL[a.lower()]
        for a in addresses
        if a.lower() in ADDRESS_TO_PROTOCOL
    }
