"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
} from "recharts";

// ── Mock Data ──────────────────────────────────────────────────────────────

const MOCK_WALLET = {
  address: "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
  ens: "vitalik.eth",
  score: 782,
  riskTier: "low" as const,
  pdEstimate: 0.183,
  scoreValidDays: 30,
  walletAge: 2847,
  activeProtocols: 12,
  totalRepaid: 4250000,
  totalBorrowed: 3890000,
  smartMoneyPercentile: 94,
};

const SHAP_FACTORS = [
  { feature: "wallet_age_days", value: 0.142, impact: "positive", interpretation: "Long on-chain history indicates reliability" },
  { feature: "repay_to_borrow_ratio", value: 0.098, impact: "positive", interpretation: "Consistent repayment behavior above 1.0x" },
  { feature: "protocols_used", value: 0.067, impact: "positive", interpretation: "Diversified protocol engagement" },
  { feature: "recent_tx_ratio", value: -0.034, impact: "negative", interpretation: "Recent activity spike warrants monitoring" },
  { feature: "aave_borrow_count", value: 0.052, impact: "positive", interpretation: "Established Aave borrowing track record" },
];

const PROTOCOL_EXPOSURE = [
  { protocol: "Aave", exposure: "$2.4M", netUsage: "+$890K", riskSignal: "Low" },
  { protocol: "Compound", exposure: "$1.1M", netUsage: "+$340K", riskSignal: "Low" },
  { protocol: "MakerDAO", exposure: "$680K", netUsage: "+$120K", riskSignal: "Medium" },
  { protocol: "Uniswap", exposure: "$450K", netUsage: "-$45K", riskSignal: "Low" },
  { protocol: "Lido", exposure: "$2.1M", netUsage: "+$1.2M", riskSignal: "Low" },
];

const ACTIVITY_HEATMAP = Array.from({ length: 90 }, (_, i) => ({
  day: i,
  value: Math.floor(Math.random() * 5),
}));

const TRANSACTION_HISTORY = [
  { month: "Jun", value: 12 },
  { month: "Jul", value: 18 },
  { month: "Aug", value: 24 },
  { month: "Sep", value: 15 },
  { month: "Oct", value: 32 },
  { month: "Nov", value: 28 },
  { month: "Dec", value: 45 },
  { month: "Jan", value: 38 },
  { month: "Feb", value: 52 },
  { month: "Mar", value: 41 },
  { month: "Apr", value: 36 },
  { month: "May", value: 48 },
];

const RECENT_TRANSACTIONS = [
  { date: "May 7, 2026", protocol: "Aave", type: "Repay", amount: "125 ETH", usd: "$312,500" },
  { date: "May 5, 2026", protocol: "Uniswap", type: "Swap", amount: "50 ETH", usd: "$125,000" },
  { date: "May 3, 2026", protocol: "Lido", type: "Stake", amount: "200 ETH", usd: "$500,000" },
  { date: "May 1, 2026", protocol: "Compound", type: "Borrow", amount: "80 ETH", usd: "$200,000" },
  { date: "Apr 28, 2026", protocol: "MakerDAO", type: "Repay", amount: "45 ETH", usd: "$112,500" },
];

// ── Semicircle Gauge ───────────────────────────────────────────────────────

function ScoreGauge({ score }: { score: number }) {
  const percentage = score / 1000;
  const angle = percentage * 180;
  
  // Risk band colors
  const bands = [
    { color: "#B91C1C", start: 0, end: 20 },
    { color: "#D97706", start: 20, end: 40 },
    { color: "#CA8A04", start: 40, end: 60 },
    { color: "#185FA5", start: 60, end: 80 },
    { color: "#15803D", start: 80, end: 100 },
  ];

  return (
    <div className="relative w-full max-w-[280px] mx-auto">
      <svg viewBox="0 0 200 120" className="w-full">
        {/* Background arc segments */}
        {bands.map((band, i) => {
          const startAngle = (band.start / 100) * 180;
          const endAngle = (band.end / 100) * 180;
          const startRad = (startAngle - 180) * (Math.PI / 180);
          const endRad = (endAngle - 180) * (Math.PI / 180);
          const r = 80;
          const cx = 100;
          const cy = 100;
          
          const x1 = cx + r * Math.cos(startRad);
          const y1 = cy + r * Math.sin(startRad);
          const x2 = cx + r * Math.cos(endRad);
          const y2 = cy + r * Math.sin(endRad);
          
          return (
            <path
              key={i}
              d={`M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`}
              fill="none"
              stroke={band.color}
              strokeWidth="12"
              strokeLinecap="butt"
              opacity={0.2}
            />
          );
        })}
        
        {/* Active arc */}
        <path
          d={`M 20 100 A 80 80 0 ${angle > 90 ? 1 : 0} 1 ${100 + 80 * Math.cos((angle - 180) * Math.PI / 180)} ${100 + 80 * Math.sin((angle - 180) * Math.PI / 180)}`}
          fill="none"
          stroke="#15803D"
          strokeWidth="12"
          strokeLinecap="round"
        />
        
        {/* Center text */}
        <text x="100" y="85" textAnchor="middle" className="font-serif text-4xl font-normal" fill="#111827">
          {score}
        </text>
        <text x="100" y="105" textAnchor="middle" className="text-xs" fill="#6B7280">
          / 1000
        </text>
      </svg>
    </div>
  );
}

// ── SHAP Feature Importance ────────────────────────────────────────────────

function ShapFeatureChart() {
  const data = SHAP_FACTORS.map(f => ({
    name: f.feature,
    value: f.value,
    impact: f.impact,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
        <XAxis 
          type="number" 
          domain={[-0.1, 0.2]}
          tick={{ fontSize: 10, fill: "#6B7280" }}
          axisLine={{ stroke: "#E5E7EB" }}
          tickLine={false}
        />
        <YAxis 
          type="category" 
          dataKey="name" 
          width={130}
          tick={{ fontSize: 10, fill: "#6B7280", fontFamily: "monospace" }}
          axisLine={false}
          tickLine={false}
        />
        <ReferenceLine x={0} stroke="#E5E7EB" />
        <Tooltip
          formatter={(v) => [Number(v).toFixed(3), "Impact"]}
          contentStyle={{
            background: "#FFFFFF",
            border: "1px solid #E5E7EB",
            borderRadius: 4,
            fontSize: 11,
          }}
        />
        <Bar dataKey="value" barSize={16}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.impact === "positive" ? "#185FA5" : "#B91C1C"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Activity Heatmap ───────────────────────────────────────────────────────

function ActivityHeatmap() {
  const weeks = 13;
  const days = 7;
  
  const getColor = (value: number) => {
    const colors = ["#F3F4F6", "#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8"];
    return colors[Math.min(value, 4)];
  };

  return (
    <div className="flex gap-1">
      {Array.from({ length: weeks }, (_, week) => (
        <div key={week} className="flex flex-col gap-1">
          {Array.from({ length: days }, (_, day) => {
            const index = week * 7 + day;
            const value = ACTIVITY_HEATMAP[index]?.value ?? 0;
            return (
              <div
                key={day}
                className="w-3 h-3 rounded-sm"
                style={{ backgroundColor: getColor(value) }}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ── Transaction Line Chart ─────────────────────────────────────────────────

function TransactionChart() {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={TRANSACTION_HISTORY} margin={{ left: 0, right: 0, top: 10, bottom: 0 }}>
        <XAxis 
          dataKey="month" 
          tick={{ fontSize: 10, fill: "#6B7280" }}
          axisLine={{ stroke: "#E5E7EB" }}
          tickLine={false}
        />
        <YAxis 
          tick={{ fontSize: 10, fill: "#6B7280" }}
          axisLine={false}
          tickLine={false}
          width={30}
        />
        <Tooltip
          contentStyle={{
            background: "#FFFFFF",
            border: "1px solid #E5E7EB",
            borderRadius: 4,
            fontSize: 11,
          }}
        />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="#185FA5" 
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Page Component ─────────────────────────────────────────────────────────

export default function Home() {
  const [address, setAddress] = useState(MOCK_WALLET.address);

  return (
    <main className="min-h-screen bg-background">
      <div className="max-w-[1400px] mx-auto px-6 py-8">
        
        {/* ── HERO / REPORT TITLE ─────────────────────────────────────── */}
        <section className="mb-10">
          <h1 className="headline-serif text-4xl md:text-5xl text-foreground mb-3">
            Wallet Credit Intelligence Report
          </h1>
          <p className="text-muted-foreground text-lg mb-6">
            On-chain credit risk analysis and behavioral scoring
          </p>
          
          {/* Analyst Note */}
          <div className="bg-card border border-border p-4 rounded card-shadow max-w-3xl">
            <div className="flex items-start gap-3">
              <div className="w-1 h-full bg-primary rounded-full shrink-0" style={{ minHeight: "40px" }} />
              <div>
                <p className="text-xs uppercase tracking-wider text-muted-foreground mb-1 font-medium">
                  Analyst Note
                </p>
                <p className="text-sm text-foreground leading-relaxed">
                  Wallet exhibits disciplined leverage behavior across Aave and Compound with low short-term liquidation probability. Consistent repayment patterns suggest institutional-grade risk management practices.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ── WALLET INPUT ────────────────────────────────────────────── */}
        <section className="mb-10">
          <div className="flex gap-3 max-w-xl">
            <input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="Enter wallet address (0x...)"
              className="flex-1 px-4 py-2.5 border border-border rounded bg-card text-sm font-mono focus:outline-none focus:ring-1 focus:ring-primary"
            />
            <button className="px-5 py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded hover:opacity-90 transition-opacity">
              Analyze
            </button>
          </div>
        </section>

        {/* ── CREDIT SCORE SECTION (Two columns) ──────────────────────── */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* LEFT: Score Gauge */}
          <div className="bg-card border border-border p-6 rounded card-shadow">
            <h2 className="text-xs uppercase tracking-wider text-muted-foreground mb-6 font-medium">
              Credit Score
            </h2>
            <ScoreGauge score={MOCK_WALLET.score} />
            <div className="flex justify-center gap-4 mt-4 text-[10px] uppercase tracking-wider text-muted-foreground">
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ backgroundColor: "#B91C1C" }} /> High</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ backgroundColor: "#D97706" }} /> Med-High</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ backgroundColor: "#CA8A04" }} /> Medium</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ backgroundColor: "#185FA5" }} /> Low</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ backgroundColor: "#15803D" }} /> V.Low</span>
            </div>
          </div>

          {/* RIGHT: KPI Cards */}
          <div className="flex flex-col gap-4">
            {/* Top row - main metrics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-card border border-border p-4 rounded card-shadow">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Risk Tier</p>
                <p className="text-xl font-medium text-positive">Low</p>
              </div>
              <div className="bg-card border border-border p-4 rounded card-shadow">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">PD Estimate</p>
                <p className="text-xl font-medium text-foreground">{(MOCK_WALLET.pdEstimate * 100).toFixed(1)}%</p>
              </div>
              <div className="bg-card border border-border p-4 rounded card-shadow">
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Valid For</p>
                <p className="text-xl font-medium text-foreground">{MOCK_WALLET.scoreValidDays}d</p>
              </div>
            </div>

            {/* Key stats table */}
            <div className="bg-card border border-border p-4 rounded card-shadow flex-1">
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-3">Key Statistics</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Wallet Age</span>
                  <span className="font-mono text-foreground">{MOCK_WALLET.walletAge.toLocaleString()} days</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Active Protocols</span>
                  <span className="font-mono text-foreground">{MOCK_WALLET.activeProtocols}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Repaid</span>
                  <span className="font-mono text-foreground">${(MOCK_WALLET.totalRepaid / 1000000).toFixed(2)}M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Borrowed</span>
                  <span className="font-mono text-foreground">${(MOCK_WALLET.totalBorrowed / 1000000).toFixed(2)}M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Smart Money Percentile</span>
                  <span className="font-mono text-primary">{MOCK_WALLET.smartMoneyPercentile}th</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── SHAP FEATURE IMPORTANCE ─────────────────────────────────── */}
        <section className="mb-10">
          <div className="bg-card border border-border p-6 rounded card-shadow">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-sm font-medium text-foreground mb-1">{"What's Driving This Score?"}</h2>
                <p className="text-xs text-muted-foreground">SHAP-based feature importance analysis</p>
              </div>
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground bg-muted px-2 py-1 rounded">
                ML Explainability
              </span>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ShapFeatureChart />
              
              <div className="space-y-3">
                {SHAP_FACTORS.map((factor, i) => (
                  <div key={i} className="flex items-start gap-3 text-sm">
                    <span 
                      className="w-2 h-2 rounded-full mt-1.5 shrink-0"
                      style={{ backgroundColor: factor.impact === "positive" ? "#185FA5" : "#B91C1C" }}
                    />
                    <div>
                      <span className="font-mono text-xs text-muted-foreground">{factor.feature}</span>
                      <p className="text-foreground">{factor.interpretation}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex gap-6 mt-6 pt-4 border-t border-border text-xs">
              <span className="flex items-center gap-2 text-muted-foreground">
                <span className="w-3 h-1 rounded" style={{ backgroundColor: "#185FA5" }} /> Positive Impact
              </span>
              <span className="flex items-center gap-2 text-muted-foreground">
                <span className="w-3 h-1 rounded" style={{ backgroundColor: "#B91C1C" }} /> Negative Impact
              </span>
            </div>
          </div>
        </section>

        {/* ── PROTOCOL EXPOSURE + ACTIVITY HEATMAP ────────────────────── */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* Protocol Exposure Table */}
          <div className="lg:col-span-2 bg-card border border-border p-6 rounded card-shadow">
            <h2 className="text-sm font-medium text-foreground mb-4">Protocol Exposure</h2>
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 pr-4">Protocol</th>
                    <th className="text-right py-2 px-4">Exposure</th>
                    <th className="text-right py-2 px-4">Net Usage</th>
                    <th className="text-right py-2 pl-4">Risk Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {PROTOCOL_EXPOSURE.map((row, i) => (
                    <tr key={i} className="border-b border-border last:border-0">
                      <td className="py-2.5 pr-4">
                        <span className="font-medium text-foreground">{row.protocol}</span>
                      </td>
                      <td className="text-right py-2.5 px-4 font-mono text-foreground">{row.exposure}</td>
                      <td className="text-right py-2.5 px-4 font-mono" style={{ color: row.netUsage.startsWith("+") ? "#15803D" : "#B91C1C" }}>
                        {row.netUsage}
                      </td>
                      <td className="text-right py-2.5 pl-4">
                        <span 
                          className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded"
                          style={{ 
                            backgroundColor: row.riskSignal === "Low" ? "#DCFCE7" : "#FEF3C7",
                            color: row.riskSignal === "Low" ? "#15803D" : "#D97706"
                          }}
                        >
                          {row.riskSignal}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Activity Heatmap */}
          <div className="bg-card border border-border p-6 rounded card-shadow">
            <h2 className="text-sm font-medium text-foreground mb-1">Activity Heatmap</h2>
            <p className="text-xs text-muted-foreground mb-4">Last 90 days</p>
            <ActivityHeatmap />
            <div className="flex items-center gap-2 mt-4 text-[10px] text-muted-foreground">
              <span>Less</span>
              <div className="flex gap-0.5">
                {["#F3F4F6", "#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8"].map((c, i) => (
                  <div key={i} className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: c }} />
                ))}
              </div>
              <span>More</span>
            </div>
          </div>
        </section>

        {/* ── TRANSACTION HISTORY CHART ───────────────────────────────── */}
        <section className="mb-10">
          <div className="bg-card border border-border p-6 rounded card-shadow">
            <h2 className="text-sm font-medium text-foreground mb-1">Transaction History</h2>
            <p className="text-xs text-muted-foreground mb-4">Monthly transaction count (12 months)</p>
            <TransactionChart />
          </div>
        </section>

        {/* ── RISK ASSESSMENT ─────────────────────────────────────────── */}
        <section className="mb-10">
          <div className="bg-card border border-border p-6 rounded card-shadow">
            <h2 className="text-sm font-medium text-foreground mb-4">Risk Assessment</h2>
            <div className="max-w-3xl">
              <p className="text-foreground leading-relaxed mb-4">
                This wallet demonstrates strong overall creditworthiness with a low probability of default. 
                The subject maintains disciplined leverage management across major DeFi protocols, with a 
                repayment-to-borrow ratio significantly above market average. Historical behavior suggests 
                institutional-grade risk awareness and capital efficiency.
              </p>
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-positive mt-2 shrink-0" />
                  <p className="text-sm text-foreground"><strong>Strong repayment behavior</strong> — Consistent debt servicing across all active positions</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-positive mt-2 shrink-0" />
                  <p className="text-sm text-foreground"><strong>Healthy on-chain history</strong> — Extended wallet age with diversified protocol usage</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-warning mt-2 shrink-0" />
                  <p className="text-sm text-foreground"><strong>Monitor short-term activity spike</strong> — Recent transaction volume elevated vs. baseline</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── RECENT ACTIVITY TABLE ───────────────────────────────────── */}
        <section className="mb-10">
          <div className="bg-card border border-border p-6 rounded card-shadow">
            <h2 className="text-sm font-medium text-foreground mb-4">Recent Activity</h2>
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 pr-4">Date</th>
                    <th className="text-left py-2 px-4">Protocol</th>
                    <th className="text-left py-2 px-4">Type</th>
                    <th className="text-right py-2 px-4">Amount</th>
                    <th className="text-right py-2 pl-4">USD Value</th>
                  </tr>
                </thead>
                <tbody>
                  {RECENT_TRANSACTIONS.map((tx, i) => (
                    <tr key={i} className="border-b border-border last:border-0">
                      <td className="py-2.5 pr-4 font-mono text-muted-foreground">{tx.date}</td>
                      <td className="py-2.5 px-4 font-medium text-foreground">{tx.protocol}</td>
                      <td className="py-2.5 px-4">
                        <span 
                          className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded"
                          style={{ 
                            backgroundColor: tx.type === "Repay" ? "#DCFCE7" : tx.type === "Borrow" ? "#FEE2E2" : "#F3F4F6",
                            color: tx.type === "Repay" ? "#15803D" : tx.type === "Borrow" ? "#B91C1C" : "#6B7280"
                          }}
                        >
                          {tx.type}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono text-foreground">{tx.amount}</td>
                      <td className="py-2.5 pl-4 text-right font-mono text-muted-foreground">{tx.usd}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── FOOTER ──────────────────────────────────────────────────── */}
        <footer className="border-t border-border pt-6 mt-10">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 text-xs text-muted-foreground">
            <div>
              <p className="font-medium text-foreground mb-1">ChainScore</p>
              <p>Institutional-grade on-chain credit intelligence</p>
            </div>
            <div className="flex gap-6">
              <a href="https://github.com/deerws/ChainScore" className="hover:text-foreground transition-colors">GitHub</a>
              <a href="https://br.linkedin.com/in/andrepinheiropaes" className="hover:text-foreground transition-colors">@andrepinheiropaes</a>
            </div>
          </div>
          <p className="text-[10px] text-muted-foreground mt-4 uppercase tracking-wider">
            For informational purposes only. Not financial advice.
          </p>
        </footer>
      </div>
    </main>
  );
}
