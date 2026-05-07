import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "About · ChainScore",
  description: "How ChainScore works — methodology, feature engineering, and credit risk modeling for DeFi wallets.",
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-4">
      <h2 className="text-xl font-bold" style={{ color: "var(--fg)" }}>{title}</h2>
      {children}
    </section>
  );
}

function MetricRow({ metric, lr, lgbm }: { metric: string; lr: string; lgbm: string }) {
  return (
    <tr className="border-b" style={{ borderColor: "var(--border)" }}>
      <td className="py-3 pr-6 text-sm font-medium" style={{ color: "var(--fg)" }}>{metric}</td>
      <td className="py-3 pr-6 text-sm text-center" style={{ color: "var(--muted)" }}>{lr}</td>
      <td className="py-3 text-sm text-center font-semibold" style={{ color: "var(--fg)" }}>{lgbm}</td>
    </tr>
  );
}

function FeatureFamily({
  name, count, description,
}: { name: string; count: number; description: string }) {
  return (
    <div className="flex gap-4 items-start p-4 rounded-xl border"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}>
      <div className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 font-bold text-white text-sm"
        style={{ background: "#185FA5" }}>
        {count}
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="font-semibold text-sm" style={{ color: "var(--fg)" }}>{name}</span>
        <span className="text-sm" style={{ color: "var(--muted)" }}>{description}</span>
      </div>
    </div>
  );
}

function Step({ n, title, description }: { n: number; title: string; description: string }) {
  return (
    <div className="flex gap-4 items-start">
      <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 font-bold text-sm text-white"
        style={{ background: "#185FA5" }}>
        {n}
      </div>
      <div className="flex flex-col gap-1 pt-0.5">
        <span className="font-semibold text-sm" style={{ color: "var(--fg)" }}>{title}</span>
        <span className="text-sm leading-relaxed" style={{ color: "var(--muted)" }}>{description}</span>
      </div>
    </div>
  );
}

export default function About() {
  return (
    <main className="flex flex-col flex-1">
      <div className="max-w-3xl mx-auto w-full px-4 py-10 sm:py-14 flex flex-col gap-12">

        {/* Hero */}
        <div className="flex flex-col gap-4">
          <h1 className="text-3xl sm:text-4xl font-bold leading-tight" style={{ color: "var(--fg)" }}>
            The Credit Score for DeFi
          </h1>
          <p className="text-base leading-relaxed" style={{ color: "var(--muted)" }}>
            DeFi lending protocols like Aave are permissionless by design — anyone can borrow,
            with no credit check. When a borrower&apos;s collateral falls below the liquidation
            threshold, the protocol automatically liquidates them, a costly outcome for all parties.
          </p>
          <p className="text-base leading-relaxed" style={{ color: "var(--muted)" }}>
            ChainScore asks: <em style={{ color: "var(--fg)" }}>can on-chain behavior predict who will be liquidated before it happens?</em>{" "}
            The answer — even at small scale — is yes. Ethereum wallets leave a rich behavioral
            trail that mirrors the signals used in traditional consumer credit scoring.
          </p>
        </div>

        {/* How it works */}
        <Section title="How It Works">
          <div className="flex flex-col gap-5">
            <Step n={1} title="Source default labels"
              description="49,748 LiquidationCall events from the Aave V2 LendingPool contract (blocks 11,500,000–25,000,000) are collected via Etherscan. Each unique liquidated borrower is labeled default = 1. Non-liquidated active borrowers are labeled default = 0." />
            <Step n={2} title="Engineer behavioral features"
              description="45 features are computed per wallet from raw transaction history: ETH flows, DeFi protocol usage, Aave borrow/repay patterns, counterparty diversity, and temporal activity regularity. These features mirror FICO-style behavioral signals." />
            <Step n={3} title="Train and calibrate the model"
              description="A LightGBM gradient boosting model is trained on a temporal split (block 17,000,000 as cutoff) to prevent data leakage. Probabilities are calibrated via Platt scaling to produce reliable PD estimates." />
            <Step n={4} title="Convert PD to a 0–1000 score"
              description="The probability of default (PD) is linearly mapped to a 0–1000 score: Score = round(1000 × (1 – PD)). Higher scores mean lower risk — intentionally mirroring FICO's direction. SHAP values identify which features drove each score." />
          </div>
        </Section>

        {/* Feature families */}
        <Section title="Feature Engineering — 45 Features">
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            Five feature families capture different dimensions of on-chain credit behavior:
          </p>
          <div className="flex flex-col gap-3">
            <FeatureFamily name="Transaction Volume" count={9}
              description="ETH sent/received, net flow, tx count, value statistics (mean, std, max), failed tx ratio" />
            <FeatureFamily name="Counterparty Graph" count={7}
              description="Unique senders/receivers, top-1 concentration, contract interaction ratio, self-transfer ratio" />
            <FeatureFamily name="Protocol Diversity" count={7}
              description="DeFi breadth (Aave, Compound, Uniswap usage), Shannon entropy index, unique ERC-20 tokens" />
            <FeatureFamily name="Collateral Behavior" count={8}
              description="Aave deposit/borrow/repay/withdraw counts, repay-to-borrow ratio, gas price percentiles" />
            <FeatureFamily name="Temporal Consistency" count={14}
              description="Wallet age, active days, inter-tx gaps, burst coefficient, dormancy periods, weekday ratio" />
          </div>
        </Section>

        {/* Results */}
        <Section title="Model Performance">
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            Results on the current 299-wallet MVP dataset (temporal split at block 17,000,000).
            Full 15,809-wallet training is in progress — LightGBM performance is expected to
            improve significantly above 1,000 samples, consistent with credit risk literature.
          </p>

          <div className="rounded-2xl border overflow-hidden"
            style={{ borderColor: "var(--border)" }}>
            <table className="w-full">
              <thead>
                <tr style={{ background: "var(--bg)" }}>
                  <th className="py-3 px-4 text-left text-xs font-semibold uppercase tracking-wide"
                    style={{ color: "var(--muted)" }}>Metric</th>
                  <th className="py-3 px-4 text-center text-xs font-semibold uppercase tracking-wide"
                    style={{ color: "var(--muted)" }}>Logistic Regression</th>
                  <th className="py-3 px-4 text-center text-xs font-semibold uppercase tracking-wide"
                    style={{ color: "var(--muted)" }}>LightGBM</th>
                </tr>
              </thead>
              <tbody style={{ background: "var(--card)" }}>
                <MetricRow metric="ROC-AUC" lr="0.613" lgbm="0.588" />
                <MetricRow metric="KS Statistic" lr="0.300" lgbm="0.233" />
                <MetricRow metric="Gini Coefficient" lr="0.227" lgbm="0.176" />
                <MetricRow metric="Brier Score" lr="0.264" lgbm="0.249" />
                <MetricRow metric="Lift @ Decile 2" lr="1.33×" lgbm="1.17×" />
              </tbody>
            </table>
          </div>

          <p className="text-xs" style={{ color: "var(--muted)" }}>
            LR outperforms LightGBM at this sample size — a well-documented pattern in credit
            risk: gradient boosting requires ≥ 1,000 labeled samples to surpass linear models.
            The KS statistic of 0.30 is meaningful separation for a prototype with no hyperparameter tuning.
          </p>
        </Section>

        {/* Risk tiers */}
        <Section title="Risk Tiers">
          <div className="rounded-2xl border overflow-hidden" style={{ borderColor: "var(--border)" }}>
            <table className="w-full">
              <thead>
                <tr style={{ background: "var(--bg)" }}>
                  {["Tier", "Score Range", "PD Range", "Interpretation"].map((h) => (
                    <th key={h} className="py-3 px-4 text-left text-xs font-semibold uppercase tracking-wide"
                      style={{ color: "var(--muted)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody style={{ background: "var(--card)" }}>
                {[
                  { tier: "Very Low",  range: "800–1000", pd: "< 20%",   note: "Investment grade — comparable to AAA–A" },
                  { tier: "Low",       range: "650–799",  pd: "20–35%",  note: "Investment grade — comparable to BBB" },
                  { tier: "Medium",    range: "500–649",  pd: "35–50%",  note: "Sub-investment — comparable to BB" },
                  { tier: "High",      range: "300–499",  pd: "50–70%",  note: "Speculative — comparable to B" },
                  { tier: "Very High", range: "0–299",    pd: "> 70%",   note: "Distressed — comparable to CCC or below" },
                ].map((row) => (
                  <tr key={row.tier} className="border-b" style={{ borderColor: "var(--border)" }}>
                    <td className="py-3 px-4 text-sm font-medium" style={{ color: "var(--fg)" }}>{row.tier}</td>
                    <td className="py-3 px-4 text-sm font-mono" style={{ color: "var(--muted)" }}>{row.range}</td>
                    <td className="py-3 px-4 text-sm" style={{ color: "var(--muted)" }}>{row.pd}</td>
                    <td className="py-3 px-4 text-sm" style={{ color: "var(--muted)" }}>{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        {/* Tech stack */}
        <Section title="Tech Stack">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              { layer: "Data Pipeline",   tools: "Python · Etherscan API · Alchemy RPC · Pandas · PyArrow" },
              { layer: "Modeling",        tools: "Scikit-learn · LightGBM · SHAP · SciPy · Statsmodels" },
              { layer: "API",             tools: "FastAPI · Pydantic · Uvicorn · Python-dotenv" },
              { layer: "Frontend",        tools: "Next.js 16 · TypeScript · Tailwind CSS · Recharts" },
              { layer: "Blockchain",      tools: "Web3.py · Solidity · Aave V2 LendingPool ABI" },
              { layer: "Infrastructure",  tools: "Git · GitHub · Bun · venv" },
            ].map(({ layer, tools }) => (
              <div key={layer} className="rounded-xl border p-4"
                style={{ background: "var(--card)", borderColor: "var(--border)" }}>
                <div className="text-xs font-semibold mb-1" style={{ color: "var(--muted)" }}>{layer}</div>
                <div className="text-sm" style={{ color: "var(--fg)" }}>{tools}</div>
              </div>
            ))}
          </div>
        </Section>

        {/* CTA */}
        <div className="rounded-2xl border p-6 sm:p-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
          style={{ background: "var(--card)", borderColor: "var(--border)" }}>
          <div className="flex flex-col gap-1">
            <span className="font-bold" style={{ color: "var(--fg)" }}>Try it live</span>
            <span className="text-sm" style={{ color: "var(--muted)" }}>
              Score any Ethereum wallet in seconds.
            </span>
          </div>
          <div className="flex gap-3">
            <Link href="/"
              className="px-5 py-2.5 rounded-xl text-white font-semibold text-sm"
              style={{ background: "#185FA5" }}>
              Score a wallet →
            </Link>
            <a href="https://github.com/deerws/ChainScore"
              target="_blank" rel="noreferrer"
              className="px-5 py-2.5 rounded-xl border font-semibold text-sm transition-opacity hover:opacity-70"
              style={{ borderColor: "var(--border)", color: "var(--fg)" }}>
              GitHub ↗
            </a>
          </div>
        </div>

        {/* Author */}
        <div className="flex flex-col gap-2 pb-4">
          <span className="text-sm font-semibold" style={{ color: "var(--fg)" }}>Built by André Pinheiro Paes</span>
          <span className="text-sm" style={{ color: "var(--muted)" }}>
            Computer Science student at UFSC. Interested in credit risk, financial markets, and AI.
          </span>
          <div className="flex gap-4 text-sm mt-1">
            <a href="https://br.linkedin.com/in/andrepinheiropaes" target="_blank" rel="noreferrer"
              className="underline hover:opacity-70" style={{ color: "var(--muted)" }}>LinkedIn</a>
            <a href="https://github.com/deerws" target="_blank" rel="noreferrer"
              className="underline hover:opacity-70" style={{ color: "var(--muted)" }}>GitHub</a>
            <a href="mailto:paes.andre33@gmail.com"
              className="underline hover:opacity-70" style={{ color: "var(--muted)" }}>Email</a>
          </div>
        </div>

      </div>
    </main>
  );
}
