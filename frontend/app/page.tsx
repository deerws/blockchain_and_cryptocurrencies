"use client";

import { useEffect, useState, useCallback } from "react";
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
  label: string; color: string; bg: string; border: string; range: string; hex: string;
}> = {
  very_low:  { label: "Very Low",  color: "text-emerald-600", bg: "bg-emerald-50  dark:bg-emerald-950", border: "border-emerald-300 dark:border-emerald-700", range: "800–1000", hex: "#10b981" },
  low:       { label: "Low",       color: "text-green-600",   bg: "bg-green-50    dark:bg-green-950",   border: "border-green-300   dark:border-green-700",   range: "650–799",  hex: "#22c55e" },
  medium:    { label: "Medium",    color: "text-yellow-600",  bg: "bg-yellow-50   dark:bg-yellow-950",  border: "border-yellow-300  dark:border-yellow-700",  range: "500–649",  hex: "#eab308" },
  high:      { label: "High",      color: "text-orange-600",  bg: "bg-orange-50   dark:bg-orange-950",  border: "border-orange-300  dark:border-orange-700",  range: "300–499",  hex: "#f97316" },
  very_high: { label: "Very High", color: "text-red-600",     bg: "bg-red-50      dark:bg-red-950",     border: "border-red-300     dark:border-red-700",     range: "0–299",    hex: "#ef4444" },
};

const TIER_ORDER: RiskTier[] = ["very_high", "high", "medium", "low", "very_low"];

// ── Theme hook ─────────────────────────────────────────────────────────────

function useTheme() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = useCallback(() => {
    setDark((d) => {
      const next = !d;
      document.documentElement.classList.toggle("dark", next);
      localStorage.setItem("theme", next ? "dark" : "light");
      return next;
    });
  }, []);

  return { dark, toggle };
}

// ── Semicircle gauge ───────────────────────────────────────────────────────

function ScoreGauge({ score, tier }: { score: number; tier: RiskTier }) {
  const R = 110;
  const cx = 150;
  const cy = 145;
  const arcLen = Math.PI * R;                          // = π × r ≈ 345.6
  const filled = (score / 1000) * arcLen;
  const offset = arcLen - filled;
  const tierHex = TIER_CONFIG[tier].hex;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg viewBox="0 0 300 165" className="w-64">
        <defs>
          <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#ef4444" />
            <stop offset="40%"  stopColor="#eab308" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>
        {/* Track */}
        <path
          d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="18"
          strokeLinecap="round"
          className="text-gray-200 dark:text-slate-700"
        />
        {/* Fill */}
        <path
          d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none"
          stroke="url(#g)"
          strokeWidth="18"
          strokeLinecap="round"
          strokeDasharray={`${arcLen}`}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
        {/* Score number */}
        <text x={cx} y={cy - 10} textAnchor="middle" className="font-bold" style={{ fill: tierHex, fontSize: 46, fontWeight: 800 }}>
          {score}
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle" style={{ fill: "#94a3b8", fontSize: 13 }}>
          out of 1000
        </text>
        {/* Labels */}
        <text x={cx - R + 4} y={cy + 22} style={{ fill: "#94a3b8", fontSize: 11 }}>0</text>
        <text x={cx + R - 16} y={cy + 22} style={{ fill: "#94a3b8", fontSize: 11 }}>1000</text>
      </svg>
    </div>
  );
}

// ── Risk spectrum bar ──────────────────────────────────────────────────────

function RiskSpectrum({ current }: { current: RiskTier }) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>Risk Spectrum</span>
      <div className="flex rounded-full overflow-hidden h-3">
        {(["very_high", "high", "medium", "low", "very_low"] as RiskTier[]).map((t) => (
          <div
            key={t}
            className="flex-1 transition-all duration-300"
            style={{
              background: TIER_CONFIG[t].hex,
              opacity: current === t ? 1 : 0.25,
              transform: current === t ? "scaleY(1.4)" : "scaleY(1)",
            }}
          />
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
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold" style={{ color: "var(--fg)" }}>
          Top Risk Factors
        </h2>
        <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "var(--border)", color: "var(--muted)" }}>
          SHAP · LightGBM
        </span>
      </div>

      <ResponsiveContainer width="100%" height={data.length * 36 + 20}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 32, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={["auto", "auto"]} tick={{ fontSize: 11, fill: "var(--muted)" }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12, fill: "var(--muted)" }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(v) => [v != null ? Number(v).toFixed(3) : "—", "SHAP value"]}
            contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "var(--fg)", fontWeight: 600 }}
          />
          <ReferenceLine x={0} stroke="var(--border)" />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
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

// ── Analyst insights card ──────────────────────────────────────────────────

function AnalystCard({ result }: { result: ScoreResponse }) {
  const pd = result.probability_of_default;
  const expectedDefaults = Math.round(pd * 100);
  const tier = TIER_CONFIG[result.risk_tier];

  const grade =
    result.score >= 800 ? "Investment Grade (AAA–A equivalent)" :
    result.score >= 650 ? "Investment Grade (BBB equivalent)" :
    result.score >= 500 ? "Sub-Investment Grade (BB equivalent)" :
    result.score >= 300 ? "Speculative Grade (B equivalent)" :
                          "Distressed / CCC equivalent";

  return (
    <div className="rounded-2xl border p-5 flex flex-col gap-4" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
      <h2 className="text-sm font-semibold" style={{ color: "var(--fg)" }}>Analyst Insights</h2>

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl p-4 flex flex-col gap-1" style={{ background: "var(--bg)" }}>
          <span className="text-xs" style={{ color: "var(--muted)" }}>Expected defaults per 100 similar wallets</span>
          <span className="text-3xl font-bold" style={{ color: TIER_CONFIG[result.risk_tier].hex }}>
            {expectedDefaults}
          </span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>out of 100</span>
        </div>

        <div className="rounded-xl p-4 flex flex-col gap-1" style={{ background: "var(--bg)" }}>
          <span className="text-xs" style={{ color: "var(--muted)" }}>Credit grade equivalent</span>
          <span className="text-sm font-semibold mt-1" style={{ color: "var(--fg)" }}>{grade}</span>
        </div>
      </div>

      <div className={`rounded-xl p-4 border ${tier.border} ${tier.bg}`}>
        <p className={`text-sm ${tier.color}`}>
          <strong>{tier.label} Risk — </strong>
          {result.risk_tier === "very_low" && "This wallet shows strong repayment patterns. PD below 20% — suitable for most DeFi lending protocols with standard collateral requirements."}
          {result.risk_tier === "low" && "Solid credit profile with low default probability. Comparable to an investment-grade borrower in traditional credit markets."}
          {result.risk_tier === "medium" && "Moderate credit risk. On-chain behavior suggests some exposure. Lenders may require higher collateral ratios or interest spreads."}
          {result.risk_tier === "high" && "Elevated default risk. This wallet has behavioral patterns consistent with historical Aave V2 liquidation events. Use with caution."}
          {result.risk_tier === "very_high" && "High probability of default. On-chain signals strongly resemble wallets that were liquidated on Aave V2. Not recommended for unsecured exposure."}
        </p>
      </div>
    </div>
  );
}

// ── Recent searches ────────────────────────────────────────────────────────

function RecentSearches({ onSelect }: { onSelect: (addr: string) => void }) {
  const [recents, setRecents] = useState<string[]>([]);

  useEffect(() => {
    try {
      setRecents(JSON.parse(localStorage.getItem("cs_recents") ?? "[]"));
    } catch { /* empty */ }
  }, []);

  if (!recents.length) return null;

  return (
    <div className="w-full max-w-2xl flex flex-col gap-2">
      <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>Recent searches</span>
      <div className="flex flex-wrap gap-2">
        {recents.map((addr) => (
          <button
            key={addr}
            onClick={() => onSelect(addr)}
            className="font-mono text-xs px-3 py-1.5 rounded-lg border transition-colors hover:border-blue-400"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--muted)" }}
          >
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
    const next = [addr, ...prev.filter((a) => a !== addr)].slice(0, 5);
    localStorage.setItem("cs_recents", JSON.stringify(next));
  } catch { /* empty */ }
}

// ── Skeleton ───────────────────────────────────────────────────────────────

function Skeleton() {
  return (
    <div className="w-full max-w-2xl flex flex-col gap-4 animate-pulse">
      <div className="rounded-2xl border p-6 flex flex-col gap-6" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <div className="flex justify-between">
          <div className="h-4 w-64 rounded" style={{ background: "var(--border)" }} />
          <div className="h-6 w-24 rounded-full" style={{ background: "var(--border)" }} />
        </div>
        <div className="h-36 w-full rounded-xl" style={{ background: "var(--border)" }} />
        <div className="grid grid-cols-3 gap-4 pt-2 border-t" style={{ borderColor: "var(--border)" }}>
          {[0,1,2].map((i) => (
            <div key={i} className="h-12 rounded" style={{ background: "var(--border)" }} />
          ))}
        </div>
      </div>
      <div className="rounded-2xl border p-6" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <div className="h-48 w-full rounded" style={{ background: "var(--border)" }} />
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function Home() {
  const { dark, toggle } = useTheme();
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
      `ChainScore for ${result.wallet_address}\nScore: ${result.score}/1000\nRisk: ${result.risk_tier.replace("_", " ")}\nPD: ${(result.probability_of_default * 100).toFixed(1)}%\nValid until: ${result.score_valid_until}`
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const tier = result ? TIER_CONFIG[result.risk_tier] : null;

  return (
    <main className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="border-b px-6 py-4 flex items-center gap-3 sticky top-0 z-10 backdrop-blur-sm"
        style={{ background: "color-mix(in srgb, var(--card) 90%, transparent)", borderColor: "var(--border)" }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: "#185FA5" }}>
          <span className="text-white font-bold text-sm">C</span>
        </div>
        <span className="font-semibold text-lg" style={{ color: "var(--fg)" }}>ChainScore</span>
        <span className="ml-2 text-xs px-2 py-0.5 rounded-full border" style={{ color: "var(--muted)", borderColor: "var(--border)" }}>
          Ethereum Mainnet · Aave V2
        </span>

        <div className="ml-auto flex items-center gap-3">
          <a
            href="https://github.com/deerws/ChainScore"
            target="_blank"
            rel="noreferrer"
            className="text-xs font-medium transition-colors hover:opacity-70"
            style={{ color: "var(--muted)" }}
          >
            GitHub ↗
          </a>
          <button
            onClick={toggle}
            className="w-9 h-9 rounded-lg border flex items-center justify-center transition-colors hover:opacity-70"
            style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--muted)" }}
            aria-label="Toggle dark mode"
          >
            {dark ? "☀️" : "🌙"}
          </button>
        </div>
      </header>

      <div className="flex-1 flex flex-col items-center px-4 py-14 gap-10">
        {/* Hero */}
        <div className="text-center max-w-2xl flex flex-col gap-4">
          <div className="inline-flex items-center gap-2 self-center px-3 py-1 rounded-full border text-xs font-medium"
            style={{ borderColor: "var(--border)", color: "var(--muted)", background: "var(--card)" }}>
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Live · Powered by LightGBM + 15k on-chain labels
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold leading-tight" style={{ color: "var(--fg)" }}>
            Credit score any<br />Ethereum wallet
          </h1>
          <p className="text-base" style={{ color: "var(--muted)" }}>
            Trained on 49,748 Aave V2 liquidation events. Returns a 0–1000 score, probability of default,
            and SHAP-driven risk factors — in seconds.
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
              placeholder="0x… Ethereum wallet address"
              className="flex-1 rounded-xl border px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
              style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--fg)" }}
            />
            <button
              onClick={() => handleScore()}
              disabled={loading}
              className="px-6 py-3 rounded-xl text-white font-semibold text-sm disabled:opacity-60 transition-opacity shrink-0"
              style={{ background: "#185FA5" }}
            >
              {loading ? "Scoring…" : "Score →"}
            </button>
          </div>
          <RecentSearches onSelect={(a) => handleScore(a)} />
        </div>

        {/* Error */}
        {error && (
          <div className="w-full max-w-2xl rounded-xl border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950 px-5 py-4 text-red-700 dark:text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && <Skeleton />}

        {/* Result */}
        {result && tier && !loading && (
          <div className="w-full max-w-2xl flex flex-col gap-4">
            {/* Score card */}
            <div className="rounded-2xl border p-6 flex flex-col gap-6 shadow-sm"
              style={{ background: "var(--card)", borderColor: "var(--border)" }}>
              {/* Header row */}
              <div className="flex items-start justify-between gap-4">
                <div className="flex flex-col gap-1 min-w-0">
                  <span className="text-xs" style={{ color: "var(--muted)" }}>Wallet</span>
                  <span className="text-sm font-mono truncate" style={{ color: "var(--fg)" }}>
                    {result.wallet_address}
                  </span>
                </div>
                <div className="flex gap-2 shrink-0">
                  <button
                    onClick={copyResult}
                    className="px-3 py-1 rounded-lg border text-xs font-medium transition-colors hover:opacity-70"
                    style={{ borderColor: "var(--border)", color: "var(--muted)", background: "var(--bg)" }}
                  >
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
              <div className="grid grid-cols-3 gap-3 pt-4 border-t" style={{ borderColor: "var(--border)" }}>
                {[
                  { label: "Prob. of Default", value: `${(result.probability_of_default * 100).toFixed(1)}%` },
                  { label: "Model",            value: result.model_version },
                  { label: "Valid until",      value: result.score_valid_until },
                ].map(({ label, value }) => (
                  <div key={label} className="flex flex-col items-center gap-1 p-3 rounded-xl"
                    style={{ background: "var(--bg)" }}>
                    <span className="text-xs" style={{ color: "var(--muted)" }}>{label}</span>
                    <span className="text-sm font-bold" style={{ color: "var(--fg)" }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Analyst insights */}
            <AnalystCard result={result} />

            {/* SHAP chart */}
            {result.top_factors.length > 0 && (
              <div className="rounded-2xl border p-6 shadow-sm"
                style={{ background: "var(--card)", borderColor: "var(--border)" }}>
                <ShapChart factors={result.top_factors} />
                <p className="text-xs mt-4" style={{ color: "var(--muted)" }}>
                  SHAP values measure each feature&apos;s contribution to the final score. Positive values push toward default; negative values push toward non-default.
                </p>
              </div>
            )}

            {/* Disclaimer */}
            <p className="text-xs text-center pb-2" style={{ color: "var(--muted)" }}>
              ChainScore is a research prototype. Not financial advice. Scores are based on historical on-chain patterns and may not reflect current wallet behavior.
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t px-6 py-4 text-center text-xs" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
        Built by{" "}
        <a href="https://br.linkedin.com/in/andrepinheiropaes" className="underline hover:opacity-70" target="_blank" rel="noreferrer">
          André Pinheiro Paes
        </a>
        {" · "}
        <a href="https://github.com/deerws/ChainScore" className="underline hover:opacity-70" target="_blank" rel="noreferrer">
          Open source
        </a>
        {" · For research and educational purposes"}
      </footer>
    </main>
  );
}
