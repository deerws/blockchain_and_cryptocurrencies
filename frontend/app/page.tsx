"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY  = process.env.NEXT_PUBLIC_API_KEY  ?? "";

// ── Types ──────────────────────────────────────────────────────────────────

type RiskTier = "very_low" | "low" | "medium" | "high" | "very_high";

interface ShapFactor {
  feature: string;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}

interface ScoreResponse {
  wallet_address: string;
  score: number;
  risk_tier: RiskTier;
  probability_of_default: number;
  top_factors: ShapFactor[];
  model_version: string;
  scored_at: string;
  score_valid_until: string;
}

// ── Config ─────────────────────────────────────────────────────────────────

const TIER_CONFIG: Record<RiskTier, {
  label: string; color: string; bg: string; border: string; hex: string;
}> = {
  very_low:  { label: "Very Low",  color: "text-emerald-600", bg: "bg-emerald-50 dark:bg-emerald-950", border: "border-emerald-300 dark:border-emerald-700", hex: "#10b981" },
  low:       { label: "Low",       color: "text-green-600",   bg: "bg-green-50   dark:bg-green-950",   border: "border-green-300   dark:border-green-700",   hex: "#22c55e" },
  medium:    { label: "Medium",    color: "text-yellow-600",  bg: "bg-yellow-50  dark:bg-yellow-950",  border: "border-yellow-300  dark:border-yellow-700",  hex: "#eab308" },
  high:      { label: "High",      color: "text-orange-600",  bg: "bg-orange-50  dark:bg-orange-950",  border: "border-orange-300  dark:border-orange-700",  hex: "#f97316" },
  very_high: { label: "Very High", color: "text-red-600",     bg: "bg-red-50     dark:bg-red-950",     border: "border-red-300     dark:border-red-700",     hex: "#ef4444" },
};

// ── Semicircle gauge ───────────────────────────────────────────────────────

function ScoreGauge({ score, tier }: { score: number; tier: RiskTier }) {
  const R = 110;
  const cx = 150;
  const cy = 145;
  const arcLen = Math.PI * R;
  const offset = arcLen - (score / 1000) * arcLen;
  const tierHex = TIER_CONFIG[tier].hex;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 300 165" className="w-56 sm:w-64">
        <defs>
          <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#ef4444" />
            <stop offset="40%"  stopColor="#eab308" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>
        <path
          d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none" stroke="currentColor" strokeWidth="18" strokeLinecap="round"
          className="text-gray-200 dark:text-slate-700"
        />
        <path
          d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none" stroke="url(#g)" strokeWidth="18" strokeLinecap="round"
          strokeDasharray={`${arcLen}`} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
        <text x={cx} y={cy - 10} textAnchor="middle"
          style={{ fill: tierHex, fontSize: 46, fontWeight: 800 }}>
          {score}
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle"
          style={{ fill: "#94a3b8", fontSize: 13 }}>
          out of 1000
        </text>
        <text x={cx - R + 4} y={cy + 22} style={{ fill: "#94a3b8", fontSize: 11 }}>0</text>
        <text x={cx + R - 16} y={cy + 22} style={{ fill: "#94a3b8", fontSize: 11 }}>1000</text>
      </svg>
    </div>
  );
}

// ── Risk spectrum ──────────────────────────────────────────────────────────

function RiskSpectrum({ current }: { current: RiskTier }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>Risk Spectrum</span>
      <div className="flex rounded-full overflow-hidden h-3">
        {(["very_high", "high", "medium", "low", "very_low"] as RiskTier[]).map((t) => (
          <div key={t} className="flex-1 transition-all duration-300" style={{
            background: TIER_CONFIG[t].hex,
            opacity: current === t ? 1 : 0.22,
            transform: current === t ? "scaleY(1.5)" : "scaleY(1)",
          }} />
        ))}
      </div>
      <div className="flex justify-between text-xs" style={{ color: "var(--muted)" }}>
        <span>Highest Risk</span>
        <span>Lowest Risk</span>
      </div>
    </div>
  );
}

// ── SHAP chart ─────────────────────────────────────────────────────────────

function ShapChart({ factors }: { factors: ShapFactor[] }) {
  const data = [...factors]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .map((f) => ({
      name: f.feature.replace(/_/g, " "),
      value: parseFloat(f.shap_value.toFixed(3)),
      direction: f.direction,
    }));

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h2 className="text-sm font-semibold" style={{ color: "var(--fg)" }}>Top Risk Factors</h2>
        <span className="text-xs px-2 py-0.5 rounded-full"
          style={{ background: "var(--border)", color: "var(--muted)" }}>
          SHAP · LightGBM
        </span>
      </div>

      <ResponsiveContainer width="100%" height={data.length * 38 + 24}>
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 28, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={["auto", "auto"]}
            tick={{ fontSize: 10, fill: "var(--muted)" }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={120}
            tick={{ fontSize: 11, fill: "var(--muted)" }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(v) => [v != null ? Number(v).toFixed(3) : "—", "SHAP value"]}
            contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "var(--fg)", fontWeight: 600 }}
          />
          <ReferenceLine x={0} stroke="var(--border)" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.direction === "increases_risk" ? "#f97316" : "#10b981"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="flex gap-4 text-xs" style={{ color: "var(--muted)" }}>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-orange-400 inline-block" /> Increases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-emerald-500 inline-block" /> Decreases risk
        </span>
      </div>
    </div>
  );
}

// ── Analyst insights ───────────────────────────────────────────────────────

function AnalystCard({ result }: { result: ScoreResponse }) {
  const pd = result.probability_of_default;
  const tier = TIER_CONFIG[result.risk_tier];

  const grade =
    result.score >= 800 ? "Investment Grade (AAA–A)" :
    result.score >= 650 ? "Investment Grade (BBB)" :
    result.score >= 500 ? "Sub-Investment Grade (BB)" :
    result.score >= 300 ? "Speculative Grade (B)" :
                          "Distressed / CCC";

  const interpretation: Record<RiskTier, string> = {
    very_low:  "Strong repayment patterns. PD below 20% — suitable for most DeFi lending protocols with standard collateral requirements.",
    low:       "Solid credit profile with low default probability. Comparable to an investment-grade borrower in traditional credit markets.",
    medium:    "Moderate credit risk. On-chain behavior suggests some exposure. Lenders may require higher collateral ratios or interest spreads.",
    high:      "Elevated default risk. Behavioral patterns consistent with historical Aave V2 liquidation events. Use with caution.",
    very_high: "High probability of default. On-chain signals strongly resemble wallets that were liquidated on Aave V2. Not recommended for unsecured exposure.",
  };

  return (
    <div className="rounded-2xl border p-5 flex flex-col gap-4"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}>
      <h2 className="text-sm font-semibold" style={{ color: "var(--fg)" }}>Analyst Insights</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="rounded-xl p-4 flex flex-col gap-1" style={{ background: "var(--bg)" }}>
          <span className="text-xs" style={{ color: "var(--muted)" }}>Expected defaults per 100 similar wallets</span>
          <span className="text-3xl font-bold" style={{ color: tier.hex }}>
            {Math.round(pd * 100)}
          </span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>out of 100</span>
        </div>

        <div className="rounded-xl p-4 flex flex-col gap-1" style={{ background: "var(--bg)" }}>
          <span className="text-xs" style={{ color: "var(--muted)" }}>Credit grade equivalent</span>
          <span className="text-sm font-semibold mt-1" style={{ color: "var(--fg)" }}>{grade}</span>
        </div>
      </div>

      <div className={`rounded-xl p-4 border ${tier.border} ${tier.bg}`}>
        <p className={`text-sm leading-relaxed ${tier.color}`}>
          <strong>{tier.label} Risk — </strong>
          {interpretation[result.risk_tier]}
        </p>
      </div>
    </div>
  );
}

// ── Recent searches ────────────────────────────────────────────────────────

function RecentSearches({ onSelect }: { onSelect: (addr: string) => void }) {
  const [recents, setRecents] = useState<string[]>([]);

  useEffect(() => {
    try { setRecents(JSON.parse(localStorage.getItem("cs_recents") ?? "[]")); }
    catch { /* noop */ }
  }, []);

  if (!recents.length) return null;

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>Recent</span>
      <div className="flex flex-wrap gap-2">
        {recents.map((addr) => (
          <button key={addr} onClick={() => onSelect(addr)}
            className="font-mono text-xs px-3 py-1.5 rounded-lg border transition-colors hover:border-blue-400"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--muted)" }}>
            {addr.slice(0, 6)}…{addr.slice(-4)}
          </button>
        ))}
      </div>
    </div>
  );
}

function saveRecent(addr: string) {
  try {
    const prev: string[] = JSON.parse(localStorage.getItem("cs_recents") ?? "[]");
    localStorage.setItem("cs_recents", JSON.stringify(
      [addr, ...prev.filter((a) => a !== addr)].slice(0, 5)
    ));
  } catch { /* noop */ }
}

// ── Skeleton ───────────────────────────────────────────────────────────────

function Skeleton() {
  return (
    <div className="w-full max-w-2xl flex flex-col gap-4 animate-pulse">
      {[160, 200].map((h) => (
        <div key={h} className="rounded-2xl border p-6" style={{ height: h, background: "var(--card)", borderColor: "var(--border)" }}>
          <div className="h-full rounded-xl" style={{ background: "var(--border)" }} />
        </div>
      ))}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function Home() {
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<ScoreResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [copied, setCopied]   = useState(false);

  async function handleScore(addr?: string) {
    const target = (addr ?? address).trim();
    if (!target.startsWith("0x") || target.length !== 42) {
      setError("Enter a valid Ethereum address (0x… 42 chars).");
      return;
    }
    if (addr) setAddress(addr);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/v1/score`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
        },
        body: JSON.stringify({ wallet_address: target, include_shap: true }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `API error ${res.status}`);
      }
      const data: ScoreResponse = await res.json();
      setResult(data);
      saveRecent(target);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error.");
    } finally {
      setLoading(false);
    }
  }

  function copyResult() {
    if (!result) return;
    navigator.clipboard.writeText(
      `ChainScore · ${result.wallet_address}\nScore: ${result.score}/1000  |  Risk: ${result.risk_tier.replace(/_/g, " ")}  |  PD: ${(result.probability_of_default * 100).toFixed(1)}%  |  Valid: ${result.score_valid_until}`
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const tier = result ? TIER_CONFIG[result.risk_tier] : null;

  return (
    <main className="flex flex-col flex-1">
      <div className="flex flex-col items-center px-4 py-10 sm:py-14 gap-8 sm:gap-10">

        {/* Hero */}
        <div className="text-center max-w-2xl flex flex-col gap-4">
          <div className="inline-flex items-center gap-2 self-center px-3 py-1 rounded-full border text-xs font-medium"
            style={{ borderColor: "var(--border)", color: "var(--muted)", background: "var(--card)" }}>
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Live · LightGBM · 49,748 Aave V2 liquidation labels
          </div>
          <h1 className="text-3xl sm:text-5xl font-bold leading-tight" style={{ color: "var(--fg)" }}>
            Credit score any<br className="hidden sm:block" /> Ethereum wallet
          </h1>
          <p className="text-sm sm:text-base" style={{ color: "var(--muted)" }}>
            Returns a 0–1000 score, probability of default, and SHAP-driven risk factors —
            powered by on-chain behavioral analysis of Aave V2 borrowers.
          </p>
        </div>

        {/* Search */}
        <div className="w-full max-w-2xl flex flex-col gap-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleScore()}
              placeholder="0x… wallet address"
              className="flex-1 min-w-0 rounded-xl border px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
              style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--fg)" }}
            />
            <button
              onClick={() => handleScore()}
              disabled={loading}
              className="px-4 sm:px-6 py-3 rounded-xl text-white font-semibold text-sm disabled:opacity-60 shrink-0"
              style={{ background: "#185FA5" }}
            >
              {loading ? "…" : "Score →"}
            </button>
          </div>
          <RecentSearches onSelect={(a) => handleScore(a)} />
        </div>

        {/* Error */}
        {error && (
          <div className="w-full max-w-2xl rounded-xl border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950 px-4 py-3 text-red-700 dark:text-red-400 text-sm">
            {error}
          </div>
        )}

        {loading && <Skeleton />}

        {/* Result */}
        {result && tier && !loading && (
          <div className="w-full max-w-2xl flex flex-col gap-4">

            {/* Score card */}
            <div className="rounded-2xl border p-4 sm:p-6 flex flex-col gap-5 shadow-sm"
              style={{ background: "var(--card)", borderColor: "var(--border)" }}>
              {/* Wallet + actions */}
              <div className="flex items-start justify-between gap-3 flex-wrap">
                <div className="flex flex-col gap-0.5 min-w-0">
                  <span className="text-xs" style={{ color: "var(--muted)" }}>Wallet</span>
                  <span className="text-xs sm:text-sm font-mono truncate max-w-[200px] sm:max-w-none"
                    style={{ color: "var(--fg)" }}>
                    {result.wallet_address}
                  </span>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button onClick={copyResult}
                    className="px-3 py-1 rounded-lg border text-xs font-medium hover:opacity-70"
                    style={{ borderColor: "var(--border)", color: "var(--muted)", background: "var(--bg)" }}>
                    {copied ? "Copied ✓" : "Copy"}
                  </button>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold border ${tier.color} ${tier.bg} ${tier.border}`}>
                    {tier.label} Risk
                  </span>
                </div>
              </div>

              <ScoreGauge score={result.score} tier={result.risk_tier} />
              <RiskSpectrum current={result.risk_tier} />

              {/* Stats */}
              <div className="grid grid-cols-3 gap-2 sm:gap-3 pt-4 border-t" style={{ borderColor: "var(--border)" }}>
                {[
                  { label: "Prob. Default", value: `${(result.probability_of_default * 100).toFixed(1)}%` },
                  { label: "Model",         value: result.model_version },
                  { label: "Valid until",   value: result.score_valid_until },
                ].map(({ label, value }) => (
                  <div key={label} className="flex flex-col items-center gap-1 p-2 sm:p-3 rounded-xl"
                    style={{ background: "var(--bg)" }}>
                    <span className="text-[10px] sm:text-xs text-center" style={{ color: "var(--muted)" }}>{label}</span>
                    <span className="text-xs sm:text-sm font-bold text-center" style={{ color: "var(--fg)" }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Analyst insights */}
            <AnalystCard result={result} />

            {/* SHAP chart */}
            {result.top_factors.length > 0 && (
              <div className="rounded-2xl border p-4 sm:p-6 shadow-sm"
                style={{ background: "var(--card)", borderColor: "var(--border)" }}>
                <ShapChart factors={result.top_factors} />
                <p className="text-xs mt-4" style={{ color: "var(--muted)" }}>
                  SHAP values measure each feature&apos;s marginal contribution to the prediction.
                  Positive = pushes toward default; negative = away from default.
                </p>
              </div>
            )}

            <p className="text-xs text-center pb-2" style={{ color: "var(--muted)" }}>
              Research prototype · Not financial advice · Based on historical on-chain patterns
            </p>
          </div>
        )}
      </div>

      <footer className="mt-auto border-t px-6 py-4 text-center text-xs"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
        Built by{" "}
        <a href="https://br.linkedin.com/in/andrepinheiropaes" className="underline hover:opacity-70"
          target="_blank" rel="noreferrer">André Pinheiro Paes</a>
        {" · "}
        <a href="https://github.com/deerws/ChainScore" className="underline hover:opacity-70"
          target="_blank" rel="noreferrer">Open source on GitHub</a>
      </footer>
    </main>
  );
}
