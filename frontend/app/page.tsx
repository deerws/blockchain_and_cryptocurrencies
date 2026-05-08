"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface ShapFactor {
  feature: string;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}

interface ApiResult {
  wallet_address: string;
  score: number;
  risk_tier: string;
  probability_of_default: number;
  top_factors: ShapFactor[];
  score_valid_until: string;
}

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

// Deterministic seed so server and client render identical values (no hydration mismatch)
function seededRand(seed: number) {
  const x = Math.sin(seed + 1) * 10000;
  return x - Math.floor(x);
}
const ACTIVITY_HEATMAP = Array.from({ length: 90 }, (_, i) => ({
  day: i,
  value: Math.floor(seededRand(i) * 5),
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
  
  // Risk band colors - work in both themes
  const bands = [
    { className: "gauge-critical", start: 0, end: 20 },
    { className: "gauge-poor", start: 20, end: 40 },
    { className: "gauge-fair", start: 40, end: 60 },
    { className: "gauge-good", start: 60, end: 80 },
    { className: "gauge-excellent", start: 80, end: 100 },
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
              stroke="currentColor"
              className={band.className}
              strokeWidth="12"
              strokeLinecap="butt"
              opacity={0.25}
            />
          );
        })}
        
        {/* Active arc */}
        <path
          d={`M 20 100 A 80 80 0 ${angle > 90 ? 1 : 0} 1 ${100 + 80 * Math.cos((angle - 180) * Math.PI / 180)} ${100 + 80 * Math.sin((angle - 180) * Math.PI / 180)}`}
          fill="none"
          stroke="currentColor"
          className="gauge-excellent"
          strokeWidth="12"
          strokeLinecap="round"
        />
        
        {/* Center text */}
        <text x="100" y="85" textAnchor="middle" className="headline-serif text-4xl" fill="currentColor">
          {score}
        </text>
        <text x="100" y="105" textAnchor="middle" className="text-xs" style={{ fill: 'var(--muted)' }}>
          / 1000
        </text>
      </svg>
    </div>
  );
}

// ── SHAP Feature Importance ────────────────────────────────────────────────

function ShapFeatureChart({ factors }: { factors: typeof SHAP_FACTORS }) {
  const data = factors.map(f => ({
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
          tick={{ fontSize: 10, fill: "var(--muted)" }}
          axisLine={{ stroke: "var(--border)" }}
          tickLine={false}
        />
        <YAxis 
          type="category" 
          dataKey="name" 
          width={130}
          tick={{ fontSize: 10, fill: "var(--muted)", fontFamily: "var(--font-mono)" }}
          axisLine={false}
          tickLine={false}
        />
        <ReferenceLine x={0} stroke="var(--border)" />
        <Tooltip
          formatter={(v) => [Number(v).toFixed(3), "Impact"]}
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            fontSize: 11,
            color: "var(--foreground)",
          }}
        />
        <Bar dataKey="value" barSize={16}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.impact === "positive" ? "var(--primary)" : "var(--negative)"} />
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
                className={`w-3 h-3 rounded-sm heatmap-${value}`}
                style={{
                  backgroundColor: value === 0 ? 'var(--border)' : undefined,
                  opacity: value === 0 ? 0.5 : 1,
                }}
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
          tick={{ fontSize: 10, fill: "var(--muted)" }}
          axisLine={{ stroke: "var(--border)" }}
          tickLine={false}
        />
        <YAxis 
          tick={{ fontSize: 10, fill: "var(--muted)" }}
          axisLine={false}
          tickLine={false}
          width={30}
        />
        <Tooltip
          contentStyle={{
            background: "var(--card)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            fontSize: 11,
            color: "var(--foreground)",
          }}
        />
        <Line 
          type="monotone" 
          dataKey="value" 
          stroke="var(--primary)" 
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
  const [apiResult, setApiResult] = useState<ApiResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const score       = apiResult?.score                    ?? MOCK_WALLET.score;
  const riskTier    = apiResult?.risk_tier                ?? MOCK_WALLET.riskTier;
  const pd          = apiResult?.probability_of_default   ?? MOCK_WALLET.pdEstimate;
  const validDays   = apiResult?.score_valid_until
    ? Math.max(0, Math.round((new Date(apiResult.score_valid_until).getTime() - Date.now()) / 86_400_000))
    : MOCK_WALLET.scoreValidDays;

  const shapData = apiResult?.top_factors.length
    ? apiResult.top_factors.map((f) => ({
        feature: f.feature,
        value: f.shap_value,
        impact: f.direction === "decreases_risk" ? "positive" : "negative",
        interpretation: f.direction === "decreases_risk"
          ? "Reduces probability of default"
          : "Increases probability of default",
      }))
    : SHAP_FACTORS;

  async function analyzeWallet() {
    const addr = address.trim();
    if (!addr.startsWith("0x") || addr.length !== 42) {
      setApiError("Enter a valid Ethereum address (42-char hex starting with 0x).");
      return;
    }
    setLoading(true);
    setApiError(null);
    setApiResult(null);
    try {
      const res = await fetch(`${API_BASE}/v1/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wallet_address: addr, include_shap: true }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail ?? `HTTP ${res.status}`);
      }
      const data: ApiResult = await res.json();
      setApiResult(data);
    } catch (e: unknown) {
      setApiError(e instanceof Error ? e.message : "Failed to reach API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen" style={{ background: 'var(--background)' }}>
      <div className="max-w-[1400px] mx-auto px-6 py-8">
        
        {/* ── HERO / REPORT TITLE ─────────────────────────────────────── */}
        <section className="mb-10">
          <div className="flex items-start justify-between gap-6">
            <div className="flex-1 max-w-3xl">
              <h1 className="headline-serif text-4xl md:text-5xl mb-2">
                Wallet Credit Intelligence Report
              </h1>
              <p className="text-lg mb-5" style={{ color: 'var(--muted)' }}>
                On-chain credit risk analysis and behavioral scoring
              </p>
              
              {/* Wallet Input */}
              <div className="flex gap-3 mb-3">
                <input
                  type="text"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && analyzeWallet()}
                  placeholder="Enter wallet address (0x...)"
                  className="flex-1 px-4 py-2.5 border rounded text-sm font-mono focus:outline-none focus:ring-1"
                  style={{
                    background: 'var(--card)',
                    borderColor: 'var(--border)',
                    color: 'var(--foreground)',
                  }}
                />
                <button
                  onClick={analyzeWallet}
                  disabled={loading}
                  className="px-5 py-2.5 text-sm font-medium rounded hover:opacity-90 transition-opacity disabled:opacity-50"
                  style={{ background: 'var(--primary)', color: '#fff' }}
                >
                  {loading ? "Scoring…" : "Analyze"}
                </button>
              </div>
              {apiError && (
                <p className="text-xs mb-3 px-1" style={{ color: 'var(--negative)' }}>{apiError}</p>
              )}
              {apiResult && (
                <p className="text-xs mb-3 px-1 font-mono" style={{ color: 'var(--muted)' }}>
                  Scored: {apiResult.wallet_address.slice(0,6)}…{apiResult.wallet_address.slice(-4)} · valid until {apiResult.score_valid_until}
                </p>
              )}
              
              {/* Analyst Note */}
              <div className="border p-4 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
                <div className="flex items-start gap-3">
                  <div className="w-1 h-full rounded-full shrink-0" style={{ minHeight: "40px", background: 'var(--primary)' }} />
                  <div>
                    <p className="text-xs uppercase tracking-wider mb-1 font-medium" style={{ color: 'var(--muted)' }}>
                      Analyst Note
                    </p>
                    <p className="text-sm leading-relaxed">
                      Wallet exhibits disciplined leverage behavior across Aave and Compound with low short-term liquidation probability. Consistent repayment patterns suggest institutional-grade risk management practices.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Hero Illustration */}
            <div className="hidden lg:flex items-center justify-center w-56 xl:w-72 shrink-0">
              <img 
                src="/hero-bull.png" 
                alt="ChainScore Bull - Wall Street meets Blockchain"
                className="w-full h-auto hero-bull"
              />
            </div>
          </div>
        </section>

        {/* ── CREDIT SCORE SECTION (Two columns) ──────────────────────── */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* LEFT: Score Gauge */}
          <div className="border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <h2 className="text-xs uppercase tracking-wider mb-6 font-medium" style={{ color: 'var(--muted)' }}>
              Credit Score
            </h2>
            <ScoreGauge score={score} />
            <div className="flex justify-center gap-4 mt-4 text-[10px] uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: 'var(--negative)' }} /> High</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: 'var(--warning)' }} /> Med-High</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: '#CA8A04' }} /> Medium</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: 'var(--primary)' }} /> Low</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-sm" style={{ background: 'var(--positive)' }} /> V.Low</span>
            </div>
          </div>

          {/* RIGHT: KPI Cards */}
          <div className="flex flex-col gap-4">
            {/* Top row - main metrics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="border p-4 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
                <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--muted)' }}>Risk Tier</p>
                <p className="text-xl font-medium capitalize" style={{ color: 'var(--positive)' }}>
                  {riskTier.replace("_", " ")}
                </p>
              </div>
              <div className="border p-4 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
                <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--muted)' }}>PD Estimate</p>
                <p className="text-xl font-medium">{(pd * 100).toFixed(1)}%</p>
              </div>
              <div className="border p-4 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
                <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--muted)' }}>Valid For</p>
                <p className="text-xl font-medium">{validDays}d</p>
              </div>
            </div>

            {/* Key stats table */}
            <div className="border p-4 rounded card-shadow flex-1" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
              <p className="text-[10px] uppercase tracking-wider mb-3" style={{ color: 'var(--muted)' }}>Key Statistics</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span style={{ color: 'var(--muted)' }}>Wallet Age</span>
                  <span className="font-mono">{MOCK_WALLET.walletAge.toLocaleString()} days</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--muted)' }}>Active Protocols</span>
                  <span className="font-mono">{MOCK_WALLET.activeProtocols}</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--muted)' }}>Total Repaid</span>
                  <span className="font-mono">${(MOCK_WALLET.totalRepaid / 1000000).toFixed(2)}M</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--muted)' }}>Total Borrowed</span>
                  <span className="font-mono">${(MOCK_WALLET.totalBorrowed / 1000000).toFixed(2)}M</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--muted)' }}>Smart Money Percentile</span>
                  <span className="font-mono" style={{ color: 'var(--primary)' }}>{MOCK_WALLET.smartMoneyPercentile}th</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── SHAP FEATURE IMPORTANCE ─────────────────────────────────── */}
        <section className="mb-10">
          <div className="border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-sm font-medium mb-1">{"What's Driving This Score?"}</h2>
                <p className="text-xs" style={{ color: 'var(--muted)' }}>SHAP-based feature importance analysis</p>
              </div>
              <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded" style={{ color: 'var(--muted)', background: 'var(--border)' }}>
                ML Explainability
              </span>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ShapFeatureChart factors={shapData} />

              <div className="space-y-3">
                {shapData.map((factor, i) => (
                  <div key={i} className="flex items-start gap-3 text-sm">
                    <span 
                      className="w-2 h-2 rounded-full mt-1.5 shrink-0"
                      style={{ background: factor.impact === "positive" ? "var(--primary)" : "var(--negative)" }}
                    />
                    <div>
                      <span className="font-mono text-xs" style={{ color: 'var(--muted)' }}>{factor.feature}</span>
                      <p>{factor.interpretation}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex gap-6 mt-6 pt-4 border-t text-xs" style={{ borderColor: 'var(--border)' }}>
              <span className="flex items-center gap-2" style={{ color: 'var(--muted)' }}>
                <span className="w-3 h-1 rounded" style={{ background: 'var(--primary)' }} /> Positive Impact
              </span>
              <span className="flex items-center gap-2" style={{ color: 'var(--muted)' }}>
                <span className="w-3 h-1 rounded" style={{ background: 'var(--negative)' }} /> Negative Impact
              </span>
            </div>
          </div>
        </section>

        {/* ── PROTOCOL EXPOSURE + ACTIVITY HEATMAP ────────────────────── */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* Protocol Exposure Table */}
          <div className="lg:col-span-2 border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <h2 className="text-sm font-medium mb-4">Protocol Exposure</h2>
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th className="text-left py-2 pr-4">Protocol</th>
                    <th className="text-right py-2 px-4">Exposure</th>
                    <th className="text-right py-2 px-4">Net Usage</th>
                    <th className="text-right py-2 pl-4">Risk Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {PROTOCOL_EXPOSURE.map((row, i) => (
                    <tr key={i} style={{ borderBottom: i < PROTOCOL_EXPOSURE.length - 1 ? '1px solid var(--border)' : 'none' }}>
                      <td className="py-2.5 pr-4">
                        <span className="font-medium">{row.protocol}</span>
                      </td>
                      <td className="text-right py-2.5 px-4 font-mono">{row.exposure}</td>
                      <td className="text-right py-2.5 px-4 font-mono" style={{ color: row.netUsage.startsWith("+") ? "var(--positive)" : "var(--negative)" }}>
                        {row.netUsage}
                      </td>
                      <td className="text-right py-2.5 pl-4">
                        <span 
                          className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded"
                          style={{ 
                            backgroundColor: row.riskSignal === "Low" ? "rgba(22, 101, 52, 0.15)" : "rgba(217, 119, 6, 0.15)",
                            color: row.riskSignal === "Low" ? "var(--positive)" : "var(--warning)"
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
          <div className="border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <h2 className="text-sm font-medium mb-1">Activity Heatmap</h2>
            <p className="text-xs mb-4" style={{ color: 'var(--muted)' }}>Last 90 days</p>
            <ActivityHeatmap />
            <div className="flex items-center gap-2 mt-4 text-[10px]" style={{ color: 'var(--muted)' }}>
              <span>Less</span>
              <div className="flex gap-0.5">
                {[0.3, 0.4, 0.6, 0.8, 1].map((o, i) => (
                  <div key={i} className="w-2.5 h-2.5 rounded-sm" style={{ background: 'var(--primary)', opacity: o }} />
                ))}
              </div>
              <span>More</span>
            </div>
          </div>
        </section>

        {/* ── TRANSACTION HISTORY CHART ───────────────────────────────── */}
        <section className="mb-10">
          <div className="border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <h2 className="text-sm font-medium mb-1">Transaction History</h2>
            <p className="text-xs mb-4" style={{ color: 'var(--muted)' }}>Monthly transaction count (12 months)</p>
            <TransactionChart />
          </div>
        </section>

        {/* ── RISK ASSESSMENT ─────────────────────────────────────────── */}
        <section className="mb-10">
          <div className="border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <h2 className="text-sm font-medium mb-4">Risk Assessment</h2>
            <div className="max-w-3xl">
              <p className="leading-relaxed mb-4">
                This wallet demonstrates strong overall creditworthiness with a low probability of default. 
                The subject maintains disciplined leverage management across major DeFi protocols, with a 
                repayment-to-borrow ratio significantly above market average. Historical behavior suggests 
                institutional-grade risk awareness and capital efficiency.
              </p>
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full mt-2 shrink-0" style={{ background: 'var(--positive)' }} />
                  <p className="text-sm"><strong>Strong repayment behavior</strong> — Consistent debt servicing across all active positions</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full mt-2 shrink-0" style={{ background: 'var(--positive)' }} />
                  <p className="text-sm"><strong>Healthy on-chain history</strong> — Extended wallet age with diversified protocol usage</p>
                </div>
                <div className="flex items-start gap-2">
                  <span className="w-1.5 h-1.5 rounded-full mt-2 shrink-0" style={{ background: 'var(--warning)' }} />
                  <p className="text-sm"><strong>Monitor short-term activity spike</strong> — Recent transaction volume elevated vs. baseline</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── RECENT ACTIVITY TABLE ───────────────────────────────────── */}
        <section className="mb-10">
          <div className="border p-6 rounded card-shadow" style={{ background: 'var(--card)', borderColor: 'var(--border)' }}>
            <h2 className="text-sm font-medium mb-4">Recent Activity</h2>
            <div className="overflow-x-auto">
              <table className="w-full data-table">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th className="text-left py-2 pr-4">Date</th>
                    <th className="text-left py-2 px-4">Protocol</th>
                    <th className="text-left py-2 px-4">Type</th>
                    <th className="text-right py-2 px-4">Amount</th>
                    <th className="text-right py-2 pl-4">USD Value</th>
                  </tr>
                </thead>
                <tbody>
                  {RECENT_TRANSACTIONS.map((tx, i) => (
                    <tr key={i} style={{ borderBottom: i < RECENT_TRANSACTIONS.length - 1 ? '1px solid var(--border)' : 'none' }}>
                      <td className="py-2.5 pr-4 font-mono" style={{ color: 'var(--muted)' }}>{tx.date}</td>
                      <td className="py-2.5 px-4 font-medium">{tx.protocol}</td>
                      <td className="py-2.5 px-4">
                        <span 
                          className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded"
                          style={{ 
                            backgroundColor: tx.type === "Repay" ? "rgba(22, 101, 52, 0.15)" : tx.type === "Borrow" ? "rgba(185, 28, 28, 0.15)" : "var(--border)",
                            color: tx.type === "Repay" ? "var(--positive)" : tx.type === "Borrow" ? "var(--negative)" : "var(--muted)"
                          }}
                        >
                          {tx.type}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono">{tx.amount}</td>
                      <td className="py-2.5 pl-4 text-right font-mono" style={{ color: 'var(--muted)' }}>{tx.usd}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* ── FOOTER ──────────────────────────────────────────────────── */}
        <footer className="border-t pt-6 mt-10" style={{ borderColor: 'var(--border)' }}>
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 text-xs" style={{ color: 'var(--muted)' }}>
            <div>
              <p className="font-medium mb-1" style={{ color: 'var(--foreground)' }}>ChainScore</p>
              <p>Institutional-grade on-chain credit intelligence</p>
            </div>
            <div className="flex gap-6">
              <a href="https://github.com/deerws/ChainScore" className="hover:opacity-70 transition-opacity">GitHub</a>
              <a href="https://br.linkedin.com/in/andrepinheiropaes" className="hover:opacity-70 transition-opacity">@andrepinheiropaes</a>
            </div>
          </div>
          <p className="text-[10px] mt-4 uppercase tracking-wider" style={{ color: 'var(--muted)' }}>
            For informational purposes only. Not financial advice.
          </p>
        </footer>
      </div>
    </main>
  );
}
