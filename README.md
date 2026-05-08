# ChainScore

> **On-chain credit scoring for DeFi wallets** — applying traditional credit risk methodology to Ethereum behavioral data.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![Blockchain](https://img.shields.io/badge/blockchain-Ethereum-3C3C3D.svg)](https://ethereum.org/)

---

## What is ChainScore?

ChainScore generates a **0–1000 credit score** for Ethereum wallets by modeling the probability of default (PD) from on-chain behavioral data. It mirrors the methodology used in traditional consumer credit scoring — feature engineering, gradient boosting, calibration, and scorecard interpretation — applied to DeFi lending data from Aave V2.

The system is delivered as a **B2B REST API**, enabling fintechs and digital banks to integrate on-chain credit intelligence into their lending products without building blockchain infrastructure themselves.

---

## Methodology

### 1. Default label definition

A wallet is labeled **default = 1** if it experienced at least one liquidation event on Aave V2 within the observation window. Non-default wallets are those with active borrowing positions that were never liquidated.

Labels are sourced from **49,748 LiquidationCall events** on the Aave V2 LendingPool contract (blocks 11,500,000–25,000,000), yielding **10,809 unique defaulted borrowers**.

### 2. Feature engineering — 45 features across 5 families

| Family | Count | Description |
|---|---|---|
| Transaction Volume | 9 | ETH flow, tx count, value statistics |
| Counterparty Graph | 7 | Diversity, concentration, contract interaction ratio |
| Protocol Diversity | 7 | DeFi breadth, Shannon entropy index, Aave/Compound/Uniswap flags |
| Collateral Behavior | 8 | Aave deposit/borrow/repay/withdraw counts, repay-to-borrow ratio, gas patterns |
| Temporal Consistency | 14 | Wallet age, activity regularity, dormancy periods, burst coefficient |

### 3. Models compared

| Model | Role |
|---|---|
| **Logistic Regression** | Interpretable baseline — FICO-style scorecard with log-odds coefficients |
| **LightGBM** | Gradient boosting — production model with SHAP explainability |

Both models are probability-calibrated via Platt scaling to produce reliable PD estimates.

### 4. Train / test split

Temporal split at block **17,000,000** (~April 2023) to prevent future data leakage — wallets with first activity before the cutoff go into training; those after go into testing. This mirrors production deployment constraints.

### 5. Results (299-wallet MVP dataset)

| Metric | Logistic Regression | LightGBM |
|---|---|---|
| **ROC-AUC** | **0.613** | 0.588 |
| **KS Statistic** | **0.300** | 0.233 |
| **Gini Coefficient** | **0.227** | 0.176 |
| Brier Score | 0.264 | **0.249** |
| Lift @ Decile 2 | **1.33** | 1.17 |

> **Note:** LR outperforms LightGBM on this dataset size, consistent with credit risk literature — gradient boosting requires ≥ 1,000 labeled samples to surpass linear models. Results will improve as the dataset scales to the full 15,809 available wallets.

---

## Project structure

```
ChainScore/
├── data/
│   ├── raw/                        Raw Ethereum event logs and wallet samples
│   └── processed/                  Labeled feature matrix (Parquet)
├── docs/                           Whitepapers, pitch deck, and project materials
├── notebooks/
│   ├── 01_data_exploration.ipynb   Liquidation dataset analysis
│   ├── 02_feature_engineering.ipynb Feature distributions and correlation analysis
│   ├── 03_model_training.ipynb     Training pipeline with comparison
│   └── 04_model_evaluation.ipynb   Full evaluation suite with all plots
├── src/
│   ├── data/
│   │   ├── ethereum_client.py      Alchemy RPC + Etherscan API wrapper
│   │   ├── liquidation_collector.py Aave V2 LiquidationCall event collector
│   │   ├── cohort_collector.py     Non-default borrower sampler
│   │   └── wallet_indexer.py       Transaction history indexer with checkpointing
│   ├── features/
│   │   ├── builder.py              45-feature pipeline
│   │   └── protocol_registry.py    Known DeFi contract addresses
│   ├── models/
│   │   ├── train.py                Training pipeline (LR + LightGBM, calibration)
│   │   ├── evaluate.py             KS, Gini, AUC, SHAP, lift, calibration plots
│   │   ├── predict.py              Real-time wallet scoring
│   │   └── scorecard.py            PD → 0–1000 score conversion
│   └── api/
│       ├── main.py                 FastAPI service
│       └── schemas.py              Pydantic request/response schemas
├── contracts/
│   └── ChainScoreAnchor.sol        Solidity contract for on-chain score anchoring
├── models/                         Trained model artifacts (generated locally)
├── reports/figures/                Evaluation plots (generated locally)
└── requirements.txt
```

---

## Getting started

### Prerequisites

- Python 3.11+
- Free [Alchemy API key](https://www.alchemy.com/) (Ethereum Mainnet)
- Free [Etherscan API key](https://etherscan.io/apis)

### Installation

```bash
git clone https://github.com/deerws/ChainScore.git
cd ChainScore

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

### Run the full pipeline

```bash
# 1. Collect liquidation events (Aave V2 default labels)
python -m src.data.liquidation_collector

# 2. Sample non-default cohort
python -m src.data.cohort_collector --max-wallets 5000

# 3. Index wallet transaction histories
python -m src.data.wallet_indexer \
    --wallet-list data/raw/wallet_sample_balanced.json \
    --output-dir  data/raw/wallets

# 4. Build feature matrix
python -m src.features.builder

# 5. Train models
python -m src.models.train

# 6. Evaluate
python -m src.models.evaluate

# 7. Score a specific wallet
python -m src.models.predict 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

### Run the API

```bash
.venv/bin/python3 -m uvicorn src.api.main:app --reload --port 8000
```

Swagger UI available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### Score a single wallet

```bash
curl -s -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"}'
```

#### Score multiple wallets (batch, up to 20)

```bash
curl -s -X POST http://localhost:8000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_addresses": [
      "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B"
    ],
    "include_shap": false
  }'
```

Response includes per-wallet scores, risk tiers, and PD; failed wallets carry an `error` field instead of aborting the whole batch.

### Run the frontend

```bash
cd frontend
bun install       # first time only
bun run dev
# → http://localhost:3000
```

The frontend expects the FastAPI service running on `http://localhost:8000`.

---

## Smart contract

`ChainScoreAnchor.sol` anchors score hashes on-chain (Sepolia testnet) for auditability. Lenders can verify a score was issued at a specific time without trusting the ChainScore server.

```solidity
// Verify a score commitment on-chain
bool valid = anchor.verifyScore(wallet, score, validUntil, modelVersion);
```

---

## Roadmap

- [x] **Phase 0** — Project scaffolding and repository structure
- [x] **Phase 1** — Data engineering: 49,748 liquidation events, 15,809 labeled wallets
- [x] **Phase 2** — Modeling: 45 features, LR + LightGBM, KS/Gini/AUC/SHAP evaluation
- [x] **Phase 3** — REST API (single + batch scoring) with Swagger docs; Next.js dashboard with dark mode, gauge, SHAP chart, analyst insights
- [ ] **Phase 4** — Scale dataset to 15,809 wallets and retrain models
- [ ] **Phase 5** — Deploy API + anchor scores on Sepolia testnet

---

## About

I built ChainScore to explore whether on-chain behavioral data contains meaningful credit signals — and whether traditional scorecard methodology transfers well to the blockchain domain. The answer, even at small scale, is yes.

**André Pinheiro Paes**
[LinkedIn](https://br.linkedin.com/in/andrepinheiropaes) · [GitHub](https://github.com/deerws) · [paes.andre33@gmail.com](mailto:paes.andre33@gmail.com)

---

## License

MIT — see [LICENSE](LICENSE) for details.
