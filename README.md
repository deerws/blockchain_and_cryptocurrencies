# ChainScore

> **On-chain credit scoring for DeFi wallets** — a research project on credit risk modeling using Ethereum behavioral data and machine learning.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-research--prototype-orange.svg)]()

---

## Motivation

The DeFi lending market holds over **USD 50 billion in total value locked**, yet every loan in this ecosystem requires overcollateralization (typically 150% or more). The reason: there is no equivalent of a credit bureau for blockchain wallets. Lenders cannot price risk, and creditworthy users are excluded from competitive borrowing.

ChainScore explores whether on-chain behavioral data can be transformed into a meaningful credit score — using methods adapted from traditional credit risk modeling (FICO-style scorecards, gradient boosting, calibration analysis) applied to Ethereum wallet history.

This repository documents the full pipeline: data collection, feature engineering, model training, evaluation, and deployment.

---

## What this project demonstrates

- **Credit risk methodology** — Probability of Default (PD) modeling with industry-standard metrics (KS, Gini, AUC, lift)
- **Data engineering at scale** — collecting and processing millions of Ethereum transactions
- **Feature engineering for behavioral data** — translating raw blockchain events into predictive signals
- **Model interpretability** — SHAP values, feature importance analysis, calibration plots
- **End-to-end deployment** — REST API + on-chain anchoring via Solidity smart contract

---

## Repository structure

```
chainscore/
├── data/                 Datasets (raw, processed, labeled)
├── notebooks/            Exploratory analysis and experiments
├── src/
│   ├── data/             Ethereum data collection
│   ├── features/         Feature engineering pipeline
│   ├── models/           Training, evaluation, inference
│   └── api/              FastAPI service
├── contracts/            Solidity smart contract for score anchoring
├── frontend/             React dashboard for demo
└── tests/                Unit and integration tests
```

---

## Methodology

### 1. Target definition

A wallet is labeled as **default = 1** if it experienced at least one liquidation event on Aave or Compound within 180 days of the observation window. Non-default wallets are those that maintained active borrowing positions during the same period without liquidation.

This proxy label has known limitations (discussed in the technical white paper) but is the closest available analog to credit default in traditional finance.

### 2. Feature engineering

Approximately 45 features grouped into five families:

| Family | Examples |
|---|---|
| Transaction volume | Total ETH transferred, average tx size, frequency |
| Counterparty graph | Number of unique counterparties, graph density, concentration |
| Protocol diversity | Number of DeFi protocols used, breadth index |
| Collateral behavior | Average collateral ratio, time-weighted ratio, volatility |
| Temporal consistency | Wallet age, gap statistics, activity regularity |

### 3. Models compared

- **Baseline:** Logistic Regression (interpretable, traditional credit scoring)
- **Advanced:** LightGBM (gradient boosting, current industry standard)

Both models are evaluated on the same hold-out set with identical metrics.

### 4. Evaluation metrics

- **Discrimination:** ROC-AUC, KS statistic, Gini coefficient
- **Calibration:** Reliability diagram, Brier score
- **Business impact:** Lift curves, score distribution by class
- **Robustness:** Cross-validation, temporal split validation

---

## Results (placeholder — to be updated)

| Metric | Logistic Regression | LightGBM |
|---|---|---|
| ROC-AUC | TBD | TBD |
| KS statistic | TBD | TBD |
| Gini | TBD | TBD |
| Brier score | TBD | TBD |

Detailed analysis in [notebooks/04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb).

---

## Getting started

### Prerequisites

- Python 3.11+
- A free Alchemy API key ([alchemy.com](https://www.alchemy.com/))
- A free Etherscan API key ([etherscan.io/apis](https://etherscan.io/apis))

### Installation

```bash
git clone https://github.com/deerws/chainscore.git
cd chainscore

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

### Quick start

```bash
# 1. Collect raw data (~30 min for sample dataset)
python -m src.data.liquidation_collector --limit 1000

# 2. Build features
python -m src.features.builder

# 3. Train model
python -m src.models.train

# 4. Evaluate
python -m src.models.evaluate
```

---

## Roadmap

- [x] **Phase 0** — Project scaffolding
- [ ] **Phase 1** — Data engineering: collect liquidation events, build wallet dataset
- [ ] **Phase 2** — Modeling: feature engineering, baseline + LightGBM, full evaluation
- [ ] **Phase 3** — API + smart contract anchoring on Sepolia testnet
- [ ] **Phase 4** — React dashboard for live demo

---

## Tech stack

**Data & ML:** Python, pandas, scikit-learn, LightGBM, SHAP
**Blockchain:** web3.py, Alchemy, Etherscan API, Solidity
**API:** FastAPI, Uvicorn
**Frontend:** React, TypeScript

---

## About

This project is part of my exploration of credit risk modeling and decentralized finance. It is not a production system. For inquiries about the methodology or potential collaboration:

**André Pinheiro Paes** — [paes.andre33@gmail.com](mailto:paes.andre33@gmail.com)
[LinkedIn](https://br.linkedin.com/in/andrepinheiropaes) · [GitHub](https://github.com/deerws)

---

## License

MIT — see [LICENSE](LICENSE) for details.
