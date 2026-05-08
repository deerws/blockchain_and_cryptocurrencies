import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Methodology | ChainScore",
  description:
    "ChainScore methodology - feature engineering, model architecture, and credit risk modeling for DeFi wallets.",
};

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <h2
        className="text-sm font-semibold uppercase tracking-wider"
        style={{ color: "var(--foreground)" }}
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

function MetricRow({
  metric,
  lr,
  lgbm,
}: {
  metric: string;
  lr: string;
  lgbm: string;
}) {
  return (
    <tr className="border-b" style={{ borderColor: "var(--border)" }}>
      <td
        className="py-3 pr-6 text-sm font-mono"
        style={{ color: "var(--foreground)" }}
      >
        {metric}
      </td>
      <td
        className="py-3 pr-6 text-sm text-center font-mono"
        style={{ color: "var(--muted)" }}
      >
        {lr}
      </td>
      <td
        className="py-3 text-sm text-center font-mono font-semibold"
        style={{ color: "var(--primary)" }}
      >
        {lgbm}
      </td>
    </tr>
  );
}

function FeatureFamily({
  name,
  count,
  description,
}: {
  name: string;
  count: number;
  description: string;
}) {
  return (
    <div
      className="flex gap-4 items-start p-4 rounded-lg border"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
      <div
        className="w-10 h-10 rounded flex items-center justify-center shrink-0 font-bold font-mono text-sm"
        style={{ background: "var(--primary)", color: "#fff" }}
      >
        {count}
      </div>
      <div className="flex flex-col gap-1">
        <span
          className="font-semibold text-sm"
          style={{ color: "var(--foreground)" }}
        >
          {name}
        </span>
        <span className="text-sm" style={{ color: "var(--muted)" }}>
          {description}
        </span>
      </div>
    </div>
  );
}

function Step({
  n,
  title,
  description,
}: {
  n: number;
  title: string;
  description: string;
}) {
  return (
    <div className="flex gap-4 items-start">
      <div
        className="w-8 h-8 rounded flex items-center justify-center shrink-0 font-bold font-mono text-sm"
        style={{ background: "var(--card)", color: "var(--primary)", border: "1px solid var(--border)" }}
      >
        {n}
      </div>
      <div className="flex flex-col gap-1 pt-0.5">
        <span
          className="font-semibold text-sm"
          style={{ color: "var(--foreground)" }}
        >
          {title}
        </span>
        <span
          className="text-sm leading-relaxed"
          style={{ color: "var(--muted)" }}
        >
          {description}
        </span>
      </div>
    </div>
  );
}

export default function About() {
  return (
    <main className="flex flex-col flex-1">
      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left Panel - Table of Contents */}
        <div
          className="lg:w-64 p-6 border-b lg:border-b-0 lg:border-r flex flex-col gap-4"
          style={{ borderColor: "var(--border)" }}
        >
          <div
            className="text-xs font-medium uppercase tracking-wider"
            style={{ color: "var(--muted)" }}
          >
            Documentation
          </div>
          <nav className="flex flex-col gap-1">
            {[
              "Overview",
              "Pipeline",
              "Features",
              "Performance",
              "Risk Tiers",
              "Tech Stack",
            ].map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase().replace(" ", "-")}`}
                className="text-sm py-1.5 transition-colors hover:text-[var(--foreground)]"
                style={{ color: "var(--muted)" }}
              >
                {item}
              </a>
            ))}
          </nav>
        </div>

        {/* Right Panel - Content */}
        <div className="flex-1 p-6 lg:p-8 overflow-auto">
          <div className="max-w-3xl flex flex-col gap-10">
            {/* Overview */}
            <div id="overview" className="flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: "var(--positive)" }}
                />
                <span
                  className="text-xs font-mono uppercase"
                  style={{ color: "var(--muted)" }}
                >
                  Research Prototype
                </span>
              </div>
              <h1
                className="text-2xl sm:text-3xl font-bold"
                style={{ color: "var(--foreground)" }}
              >
                On-Chain Credit Intelligence
              </h1>
              <p
                className="text-sm leading-relaxed"
                style={{ color: "var(--muted)" }}
              >
                DeFi lending protocols like Aave are permissionless by design -
                anyone can borrow, with no credit check. When a borrower&apos;s
                collateral falls below the liquidation threshold, the protocol
                automatically liquidates them.
              </p>
              <p
                className="text-sm leading-relaxed"
                style={{ color: "var(--muted)" }}
              >
                ChainScore asks:{" "}
                <em style={{ color: "var(--foreground)" }}>
                  can on-chain behavior predict who will be liquidated before it
                  happens?
                </em>{" "}
                Ethereum wallets leave a rich behavioral trail that mirrors the
                signals used in traditional consumer credit scoring.
              </p>
            </div>

            {/* Pipeline */}
            <Section title="Pipeline">
              <div className="flex flex-col gap-4">
                <Step
                  n={1}
                  title="Source Default Labels"
                  description="49,748 LiquidationCall events from the Aave V2 LendingPool contract (blocks 11,500,000-25,000,000). Each unique liquidated borrower is labeled default = 1."
                />
                <Step
                  n={2}
                  title="Engineer Behavioral Features"
                  description="45 features computed per wallet: ETH flows, DeFi protocol usage, Aave patterns, counterparty diversity, temporal activity. FICO-style behavioral signals."
                />
                <Step
                  n={3}
                  title="Train and Calibrate Model"
                  description="LightGBM gradient boosting trained on temporal split (block 17,000,000 cutoff). Probabilities calibrated via Platt scaling for reliable PD estimates."
                />
                <Step
                  n={4}
                  title="Convert PD to Score"
                  description="Probability of default mapped to 0-1000 score: Score = round(1000 x (1 - PD)). Higher scores = lower risk. SHAP values identify driving features."
                />
              </div>
            </Section>

            {/* Features */}
            <Section title="Feature Engineering - 45 Features">
              <div className="flex flex-col gap-3">
                <FeatureFamily
                  name="Transaction Volume"
                  count={9}
                  description="ETH sent/received, net flow, tx count, value statistics (mean, std, max), failed tx ratio"
                />
                <FeatureFamily
                  name="Counterparty Graph"
                  count={7}
                  description="Unique senders/receivers, top-1 concentration, contract interaction ratio, self-transfer ratio"
                />
                <FeatureFamily
                  name="Protocol Diversity"
                  count={7}
                  description="DeFi breadth (Aave, Compound, Uniswap usage), Shannon entropy index, unique ERC-20 tokens"
                />
                <FeatureFamily
                  name="Collateral Behavior"
                  count={8}
                  description="Aave deposit/borrow/repay/withdraw counts, repay-to-borrow ratio, gas price percentiles"
                />
                <FeatureFamily
                  name="Temporal Consistency"
                  count={14}
                  description="Wallet age, active days, inter-tx gaps, burst coefficient, dormancy periods, weekday ratio"
                />
              </div>
            </Section>

            {/* Performance */}
            <Section title="Model Performance">
              <p className="text-sm" style={{ color: "var(--muted)" }}>
                Results on current 299-wallet MVP dataset (temporal split at
                block 17,000,000). Full 15,809-wallet training in progress.
              </p>
              <div
                className="rounded-lg border overflow-hidden"
                style={{ borderColor: "var(--border)" }}
              >
                <table className="w-full">
                  <thead>
                    <tr style={{ background: "var(--card)" }}>
                      <th
                        className="py-3 px-4 text-left text-xs font-semibold uppercase tracking-wider font-mono"
                        style={{ color: "var(--muted)" }}
                      >
                        Metric
                      </th>
                      <th
                        className="py-3 px-4 text-center text-xs font-semibold uppercase tracking-wider font-mono"
                        style={{ color: "var(--muted)" }}
                      >
                        LogReg
                      </th>
                      <th
                        className="py-3 px-4 text-center text-xs font-semibold uppercase tracking-wider font-mono"
                        style={{ color: "var(--muted)" }}
                      >
                        LightGBM
                      </th>
                    </tr>
                  </thead>
                  <tbody style={{ background: "var(--background)" }}>
                    <MetricRow metric="ROC-AUC" lr="0.613" lgbm="0.588" />
                    <MetricRow metric="KS Statistic" lr="0.300" lgbm="0.233" />
                    <MetricRow metric="Gini" lr="0.227" lgbm="0.176" />
                    <MetricRow metric="Brier Score" lr="0.264" lgbm="0.249" />
                    <MetricRow metric="Lift @ D2" lr="1.33x" lgbm="1.17x" />
                  </tbody>
                </table>
              </div>
            </Section>

            {/* Risk Tiers */}
            <Section title="Risk Tiers">
              <div
                className="rounded-lg border overflow-hidden"
                style={{ borderColor: "var(--border)" }}
              >
                <table className="w-full">
                  <thead>
                    <tr style={{ background: "var(--card)" }}>
                      {["Tier", "Score", "PD", "Grade"].map((h) => (
                        <th
                          key={h}
                          className="py-3 px-4 text-left text-xs font-semibold uppercase tracking-wider font-mono"
                          style={{ color: "var(--muted)" }}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody style={{ background: "var(--background)" }}>
                    {[
                      { tier: "Very Low", range: "800-1000", pd: "<20%", grade: "AAA-A" },
                      { tier: "Low", range: "650-799", pd: "20-35%", grade: "BBB" },
                      { tier: "Medium", range: "500-649", pd: "35-50%", grade: "BB" },
                      { tier: "High", range: "300-499", pd: "50-70%", grade: "B" },
                      { tier: "Very High", range: "0-299", pd: ">70%", grade: "CCC" },
                    ].map((row) => (
                      <tr
                        key={row.tier}
                        className="border-b"
                        style={{ borderColor: "var(--border)" }}
                      >
                        <td
                          className="py-3 px-4 text-sm font-medium"
                          style={{ color: "var(--foreground)" }}
                        >
                          {row.tier}
                        </td>
                        <td
                          className="py-3 px-4 text-sm font-mono"
                          style={{ color: "var(--muted)" }}
                        >
                          {row.range}
                        </td>
                        <td
                          className="py-3 px-4 text-sm font-mono"
                          style={{ color: "var(--muted)" }}
                        >
                          {row.pd}
                        </td>
                        <td
                          className="py-3 px-4 text-sm font-mono"
                          style={{ color: "var(--primary)" }}
                        >
                          {row.grade}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>

            {/* Tech Stack */}
            <Section title="Tech Stack">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {[
                  { layer: "Data", tools: "Python, Etherscan API, Alchemy RPC, Pandas" },
                  { layer: "Model", tools: "Scikit-learn, LightGBM, SHAP" },
                  { layer: "API", tools: "FastAPI, Pydantic, Uvicorn" },
                  { layer: "Frontend", tools: "Next.js 16, TypeScript, Tailwind" },
                  { layer: "Chain", tools: "Web3.py, Aave V2 LendingPool ABI" },
                  { layer: "Infra", tools: "Git, GitHub, Bun" },
                ].map(({ layer, tools }) => (
                  <div
                    key={layer}
                    className="rounded-lg border p-4"
                    style={{
                      background: "var(--card)",
                      borderColor: "var(--border)",
                    }}
                  >
                    <div
                      className="text-xs font-semibold uppercase tracking-wider mb-1 font-mono"
                      style={{ color: "var(--muted)" }}
                    >
                      {layer}
                    </div>
                    <div
                      className="text-sm"
                      style={{ color: "var(--foreground)" }}
                    >
                      {tools}
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            {/* CTA */}
            <div
              className="rounded-lg border p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
              style={{
                background: "var(--card)",
                borderColor: "var(--border)",
              }}
            >
              <div className="flex flex-col gap-1">
                <span
                  className="font-semibold"
                  style={{ color: "var(--foreground)" }}
                >
                  Run Analysis
                </span>
                <span className="text-sm" style={{ color: "var(--muted)" }}>
                  Score any Ethereum wallet in seconds.
                </span>
              </div>
              <div className="flex gap-3">
                <Link
                  href="/"
                  className="px-4 py-2 rounded text-sm font-semibold"
                  style={{ background: "var(--primary)", color: "#fff" }}
                >
                  Open Terminal
                </Link>
                <a
                  href="https://github.com/deerws/ChainScore"
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 rounded border text-sm font-semibold"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--foreground)",
                  }}
                >
                  GitHub
                </a>
              </div>
            </div>

            {/* Author */}
            <div
              className="flex flex-col gap-2 py-4 border-t"
              style={{ borderColor: "var(--border)" }}
            >
              <span
                className="text-sm font-semibold"
                style={{ color: "var(--foreground)" }}
              >
                Built by Andre Pinheiro Paes
              </span>
              <span className="text-sm" style={{ color: "var(--muted)" }}>
                Computer Science student at UFSC. Interested in credit risk,
                financial markets, and AI.
              </span>
              <div className="flex gap-4 text-sm mt-1 font-mono">
                <a
                  href="https://br.linkedin.com/in/andrepinheiropaes"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:underline"
                  style={{ color: "var(--muted)" }}
                >
                  LinkedIn
                </a>
                <a
                  href="https://github.com/deerws"
                  target="_blank"
                  rel="noreferrer"
                  className="hover:underline"
                  style={{ color: "var(--muted)" }}
                >
                  GitHub
                </a>
                <a
                  href="mailto:paes.andre33@gmail.com"
                  className="hover:underline"
                  style={{ color: "var(--muted)" }}
                >
                  Email
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
