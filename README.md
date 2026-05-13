# ChainScore

> **On-chain credit scoring for DeFi wallets** — applying traditional credit risk methodology to Ethereum behavioral data.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

---

## What is ChainScore?

DeFi lending protocols like Aave are **permissionless** — anyone can borrow crypto without a credit check. When a borrower's collateral drops below the liquidation threshold, the protocol force-sells their assets, a costly outcome for everyone involved.

ChainScore asks: **can on-chain behavior predict who will be liquidated before it happens?**

The answer is yes. Ethereum wallets leave a rich behavioral trail — transaction patterns, protocol usage, repayment habits — that mirrors the signals used in traditional consumer credit scoring. ChainScore turns that trail into a **0–1000 credit score** and a calibrated **probability of default (PD)**, delivered via a REST API.

---

## For non-technical readers: how it works

Think of ChainScore like a FICO score, but for crypto wallets instead of bank accounts.

**1. Define "default"**
A borrower "defaults" when Aave forcibly liquidates their position. We collected 49,748 such liquidation events from Ethereum's history — each one is a labeled data point: *this wallet failed*.

**2. Build behavioral features**
For each wallet we compute 45 features from its raw transaction history: How active is it? Does it interact with multiple DeFi protocols? Does it repay borrowed funds consistently? How long has it existed? These are the same questions a loan officer would ask — just answered from public blockchain data instead of bank statements.

**3. Train a model**
A gradient boosting model (LightGBM) learns which behavioral patterns predict liquidation. It's trained on a historical split and tested on data it has never seen, using the same temporal holdout approach used in production credit risk systems.

**4. Score**
The model's predicted probability of default (PD) is converted to a 0–1000 scale — higher is safer, mirroring FICO's direction. Each score comes with SHAP explanations identifying *which behaviors* drove the score up or down.

---

## Dashboard

Institutional-style analyst interface built with Next.js — dark/light mode, live API integration, and a full credit report layout:

- **Score gauge** — semicircle with color-coded risk bands
- **KPI cards** — risk tier, PD estimate, score validity window
- **SHAP chart** — horizontal bar chart showing which features pushed the score up or down
- **Protocol exposure table** — Aave, Compound, Uniswap, MakerDAO, Lido
- **Activity heatmap** — 90-day on-chain activity grid
- **Transaction history** — 12-month line chart
- **Risk assessment** — analyst-style narrative with bullet-point findings

> *Screenshot coming — run locally with `bun run dev` to see the full UI.*

---

## Model performance

### Results — 8,800-wallet dataset

| Metric | Logistic Regression | LightGBM |
|---|:---:|:---:|
| **ROC-AUC** | **0.671** | 0.654 |
| **KS Statistic** | **0.328** | 0.260 |
| **Gini Coefficient** | **0.343** | 0.308 |
| Brier Score | 0.253 | **0.245** |
| Lift @ Decile 1 | **1.69×** | 1.37× |

Logistic Regression continues to lead on rank-ordering metrics (AUC, KS, Gini), while LightGBM edges it on calibration (Brier score) — consistent with credit risk literature. A KS Statistic of 0.33 crosses the threshold for scorecard-grade separation without any hyperparameter tuning. Dataset: 8,800 wallets with transaction history (5,402 defaulted, 3,398 non-default), indexed from 15,809 labeled wallets.

### Evaluation plots

| | |
|---|---|
| ![ROC curves](reports/figures/roc_curves.png) | ![KS plot](reports/figures/ks_plot.png) |
| ![Lift chart](reports/figures/lift_chart.png) | ![Calibration](reports/figures/calibration_plot.png) |
| ![SHAP summary](reports/figures/shap_summary.png) | ![Score distribution](reports/figures/score_distribution.png) |

---

## Methodology

### Feature engineering — 47 features across 5 families

| Family | Count | What it captures |
|---|:---:|---|
| **Transaction Volume** | 9 | ETH sent/received, net flow, tx count, value statistics (mean, std, max), failed tx ratio |
| **Counterparty Graph** | 7 | Unique senders/receivers, top-1 concentration, contract interaction ratio, self-transfer ratio |
| **Protocol Diversity** | 7 | DeFi breadth (Aave, Compound, Uniswap), Shannon entropy index, unique ERC-20 tokens |
| **Collateral Behavior** | 8 | Aave deposit/borrow/repay/withdraw counts, repay-to-borrow ratio, gas price percentiles |
| **Temporal Consistency** | 14 | Wallet age, active days, inter-tx gaps, burst coefficient, dormancy periods, weekday ratio |

### Risk tiers

| Tier | Score | PD range | Credit analogue |
|---|:---:|:---:|---|
| Very Low | 800–1000 | < 20% | AAA – A |
| Low | 650–799 | 20–35% | BBB |
| Medium | 500–649 | 35–50% | BB |
| High | 300–499 | 50–70% | B |
| Very High | 0–299 | > 70% | CCC or below |

### Data pipeline

- **Default labels**: 49,748 `LiquidationCall` events from the Aave V2 LendingPool contract (blocks 11,500,000 – 25,000,000), yielding 10,809 unique liquidated borrowers.
- **Non-default cohort**: Active borrowers with no liquidation history sampled from the same observation window.
- **Train/test split**: Temporal cutoff at block 17,000,000 (~April 2023) to prevent data leakage. Wallets with first activity before the cutoff go into training; those after go into testing. This mirrors production deployment constraints.
- **Calibration**: Both models are probability-calibrated via Platt scaling to produce reliable PD estimates, not just rank orderings.

---

## Project structure

```
ChainScore/
├── data/
│   ├── raw/                        Raw Ethereum event logs and wallet samples
│   └── processed/                  Labeled feature matrix (Parquet)
├── docs/                           Whitepapers, pitch deck, and project materials
├── frontend/                       Next.js dashboard (dark mode, gauge, SHAP chart)
│   └── app/
│       ├── page.tsx                Score page — main dashboard
│       ├── about/page.tsx          Methodology page for readers
│       └── components/Nav.tsx      Shared navigation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── reports/figures/                Evaluation plots (PNG)
├── src/
│   ├── data/
│   │   ├── ethereum_client.py      Alchemy RPC + Etherscan API wrapper
│   │   ├── liquidation_collector.py Aave V2 LiquidationCall event collector
│   │   ├── cohort_collector.py     Non-default borrower sampler
│   │   └── wallet_indexer.py       Transaction history indexer with checkpointing
│   ├── features/
│   │   ├── builder.py              47-feature pipeline
│   │   └── protocol_registry.py    Known DeFi contract addresses
│   ├── models/
│   │   ├── train.py                Training pipeline (LR + LightGBM, calibration)
│   │   ├── evaluate.py             KS, Gini, AUC, SHAP, lift, calibration plots
│   │   ├── predict.py              Real-time wallet scoring
│   │   └── scorecard.py            PD → 0–1000 score conversion
│   └── api/
│       ├── main.py                 FastAPI service (single + batch scoring)
│       └── schemas.py              Pydantic request/response schemas
└── contracts/
    └── ChainScoreAnchor.sol        Solidity contract for on-chain score anchoring
```

---

## Getting started

### Prerequisites

- Python 3.11+
- [Bun](https://bun.sh/) (or Node 18+ with npm)
- Free [Alchemy API key](https://www.alchemy.com/) — Ethereum Mainnet
- Free [Etherscan API key](https://etherscan.io/apis)

### Installation

```bash
git clone https://github.com/deerws/ChainScore.git
cd ChainScore

# Python environment
/path/to/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Add ALCHEMY_API_KEY, ETHERSCAN_API_KEY, and optionally API_KEY_SECRET
```

### Run the full training pipeline

```bash
# 1. Collect Aave V2 liquidation events (default labels)
python -m src.data.liquidation_collector

# 2. Sample non-default cohort
python -m src.data.cohort_collector --max-wallets 5000

# 3. Index wallet transaction histories (supports checkpointing)
python -m src.data.wallet_indexer \
    --wallet-list data/raw/wallet_sample_balanced.json \
    --output-dir  data/raw/wallets

# 4. Build feature matrix
python -m src.features.builder

# 5. Train and calibrate models
python -m src.models.train

# 6. Evaluate — generates all plots in reports/figures/
python -m src.models.evaluate

# 7. Score a wallet from the CLI
python -m src.models.predict 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

### Run the API

```bash
.venv/bin/python3 -m uvicorn src.api.main:app --reload --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

**Score a single wallet**

```bash
curl -s -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "include_shap": true}'
```

**Score multiple wallets (batch, up to 20)**

```bash
curl -s -X POST http://localhost:8000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "wallet_addresses": [
      "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B"
    ]
  }'
```

Failed wallets return an `error` field instead of aborting the whole batch. Repeated calls for the same wallet are served from a **30-minute in-memory cache** — zero extra Etherscan requests, ~50ms response time on cache hits.

### Run the frontend dashboard

```bash
cd frontend
bun install       # first time only
bun run dev
# → http://localhost:3000
```

Requires the API running on `http://localhost:8000`.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status + cache stats |
| `POST` | `/v1/score` | Score a single wallet (SHAP included) |
| `GET` | `/v1/score/{address}` | Same via GET for quick browser tests |
| `POST` | `/v1/batch` | Score up to 20 wallets in one call |
| `POST` | `/v1/portfolio` | Aggregate risk for up to 100 wallets |

**Portfolio response** returns credit-desk vocabulary: average PD, **VaR 95%** (PD at the 95th-percentile wallet), **CVaR 95% / Expected Shortfall** (mean PD of the worst 5%), tier breakdown, and high-risk concentration ratio.

```bash
curl -s -X POST http://localhost:8000/v1/portfolio \
  -H "Content-Type: application/json" \
  -d '{"wallet_addresses": ["0xabc...", "0xdef...", "0x123..."], "name": "Treasury exposure"}'
```

All endpoints are documented in Swagger UI at `http://localhost:8000/docs`. Repeated calls for the same wallet are served from a **30-min in-memory cache** (~50ms on cache hits, zero Etherscan requests).

---

## Backtesting

Beyond standard metrics, the evaluation suite includes two backtesting analyses:

**Decile hit rate** — wallets are ranked by predicted risk and split into 10 bands. For each band, we measure what fraction actually defaulted. A model with real discrimination shows monotonically decreasing hit rates from decile 1 (highest predicted risk) to decile 10.

**Precision-Recall curve** — more informative than ROC for imbalanced datasets. The default rate in DeFi lending is low, so PR curves reveal whether the model is genuinely useful at high-precision operating points, not just good at ranking.

Both plots are generated automatically by `python -m src.models.evaluate` and saved to `reports/figures/`.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data pipeline | Python · Etherscan API · Alchemy RPC · Pandas · PyArrow |
| Modeling | Scikit-learn · LightGBM · SHAP · SciPy · Statsmodels |
| API | FastAPI · Pydantic v2 · Uvicorn |
| Frontend | Next.js 16 · TypeScript · Tailwind CSS v4 · Recharts · Inter + JetBrains Mono |
| Blockchain | Web3.py · Aave V2 LendingPool ABI |

---

## Smart contract

`ChainScoreAnchor.sol` anchors score hashes on-chain (Sepolia testnet) for auditability. Lenders can verify a score was issued at a specific time without trusting the ChainScore server.

```solidity
bool valid = anchor.verifyScore(wallet, score, validUntil, modelVersion);
```

---

## Roadmap

- [x] **Phase 0** — Project scaffolding and repository structure
- [x] **Phase 1** — Data engineering: 49,748 liquidation events, 15,809 labeled wallets
- [x] **Phase 2** — Modeling: 45 features, LR + LightGBM, KS/Gini/AUC/SHAP evaluation
- [x] **Phase 3** — REST API (single + batch + portfolio scoring, Swagger, 30-min cache); Next.js institutional dashboard
- [x] **Phase 3.5** — Backtesting suite: PR curve, decile hit rate analysis, CVaR/VaR portfolio metrics
- [x] **Phase 4** — Scale to 8,800 wallets with transaction data and retrain — LR ROC-AUC 0.671, KS 0.328, Gini 0.343
- [ ] **Phase 5** — Deploy API + anchor scores on Sepolia testnet

---

## About

I've been interested in both credit risk and crypto for a while, and at some point the question became obvious: Aave borrowers have a full repayment history sitting on a public ledger — why isn't anyone turning that into a credit score? ChainScore is my attempt to answer that. The methodology is standard scorecard work (feature engineering, gradient boosting, Platt calibration, KS/Gini evaluation) applied to on-chain data instead of bank statements. The answer is yes: a KS Statistic of 0.33 and Gini of 0.34 with no hyperparameter tuning and 8,800 training samples puts this squarely in scorecard-grade territory.

**André Pinheiro Paes** — Computer Science student at UFSC, interested in credit risk, financial markets, and AI.

[LinkedIn](https://br.linkedin.com/in/andrepinheiropaes) · [GitHub](https://github.com/deerws) · [paes.andre33@gmail.com](mailto:paes.andre33@gmail.com)

---

## License

MIT — see [LICENSE](LICENSE) for details.
