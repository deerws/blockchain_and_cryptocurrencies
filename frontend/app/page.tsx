"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

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

const TIER_CONFIG: Record<
  RiskTier,
  { label: string; color: string; hex: string; grade: string }
> = {
  very_low: { label: "VERY LOW", color: "text-emerald-400", hex: "#34d399", grade: "AAA" },
  low: { label: "LOW", color: "text-green-400", hex: "#4ade80", grade: "AA" },
  medium: { label: "MEDIUM", color: "text-yellow-400", hex: "#facc15", grade: "BBB" },
  high: { label: "HIGH", color: "text-orange-400", hex: "#fb923c", grade: "BB" },
  very_high: { label: "VERY HIGH", color: "text-red-400", hex: "#f87171", grade: "CCC" },
};

// ── Score Display ──────────────────────────────────────────────────────────

function ScoreDisplay({ score, tier }: { score: number; tier: RiskTier }) {
  const config = TIER_CONFIG[tier];
  const percentage = (score / 1000) * 100;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-baseline gap-3">
        <span
          className="text-6xl sm:text-7xl font-bold font-mono tracking-tighter"
          style={{ color: config.hex }}
        >
          {score}
        </span>
        <span className="text-lg font-mono" style={{ color: "var(--muted)" }}>
          / 1000
        </span>
      </div>

      {/* Progress bar */}
      <div className="flex flex-col gap-2">
        <div
          className="h-2 rounded-full overflow-hidden"
          style={{ background: "var(--border)" }}
        >
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${percentage}%`,
              background: `linear-gradient(90deg, ${TIER_CONFIG.very_high.hex}, ${TIER_CONFIG.medium.hex}, ${TIER_CONFIG.very_low.hex})`,
            }}
          />
        </div>
        <div
          className="flex justify-between text-xs font-mono"
          style={{ color: "var(--muted)" }}
        >
          <span>0</span>
          <span>500</span>
          <span>1000</span>
        </div>
      </div>
    </div>
  );
}

// ── Risk Badge ─────────────────────────────────────────────────────────────

function RiskBadge({ tier }: { tier: RiskTier }) {
  const config = TIER_CONFIG[tier];
  return (
    <div
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded"
      style={{
        background: `${config.hex}15`,
        border: `1px solid ${config.hex}40`,
      }}
    >
      <span
        className="w-2 h-2 rounded-full"
        style={{ background: config.hex }}
      />
      <span className="text-xs font-semibold font-mono" style={{ color: config.hex }}>
        {config.label} RISK
      </span>
    </div>
  );
}

// ── Metric Card ────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  subtext,
  highlight,
}: {
  label: string;
  value: string;
  subtext?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className="p-4 rounded-lg border"
      style={{
        background: highlight ? "var(--card)" : "transparent",
        borderColor: "var(--border)",
      }}
    >
      <div
        className="text-xs font-medium uppercase tracking-wider mb-2"
        style={{ color: "var(--muted)" }}
      >
        {label}
      </div>
      <div
        className="text-xl font-bold font-mono"
        style={{ color: highlight ? "var(--primary)" : "var(--foreground)" }}
      >
        {value}
      </div>
      {subtext && (
        <div className="text-xs font-mono mt-1" style={{ color: "var(--muted)" }}>
          {subtext}
        </div>
      )}
    </div>
  );
}

// ── SHAP Chart ─────────────────────────────────────────────────────────────

function ShapChart({ factors }: { factors: ShapFactor[] }) {
  const data = [...factors]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .map((f) => ({
      name: f.feature.replace(/_/g, " ").toUpperCase(),
      value: parseFloat(f.shap_value.toFixed(3)),
      direction: f.direction,
    }));

  return (
    <div className="flex flex-col gap-4">
      <ResponsiveContainer width="100%" height={data.length * 36 + 24}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 8, right: 32, top: 4, bottom: 4 }}
        >
          <XAxis
            type="number"
            domain={["auto", "auto"]}
            tick={{ fontSize: 10, fill: "var(--muted)" }}
            axisLine={{ stroke: "var(--border)" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fontSize: 10, fill: "var(--muted)", fontFamily: "var(--font-mono)" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(v) => [v != null ? Number(v).toFixed(3) : "-", "SHAP"]}
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              fontSize: 11,
              fontFamily: "var(--font-mono)",
            }}
            labelStyle={{ color: "var(--foreground)", fontWeight: 600 }}
          />
          <ReferenceLine x={0} stroke="var(--border)" />
          <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={14}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.direction === "increases_risk" ? "#f87171" : "#34d399"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div
        className="flex gap-6 text-xs font-mono"
        style={{ color: "var(--muted)" }}
      >
        <span className="flex items-center gap-2">
          <span className="w-3 h-1 rounded bg-red-400" /> RISK INCREASE
        </span>
        <span className="flex items-center gap-2">
          <span className="w-3 h-1 rounded bg-emerald-400" /> RISK DECREASE
        </span>
      </div>
    </div>
  );
}

// ── Recent Searches ────────────────────────────────────────────────────────

function RecentSearches({ onSelect }: { onSelect: (addr: string) => void }) {
  const [recents, setRecents] = useState<string[]>([]);

  useEffect(() => {
    try {
      setRecents(JSON.parse(localStorage.getItem("cs_recents") ?? "[]"));
    } catch {
      /* noop */
    }
  }, []);

  if (!recents.length) return null;

  return (
    <div className="flex flex-col gap-2">
      <span
        className="text-xs font-medium uppercase tracking-wider"
        style={{ color: "var(--muted)" }}
      >
        Recent Queries
      </span>
      <div className="flex flex-wrap gap-2">
        {recents.map((addr) => (
          <button
            key={addr}
            onClick={() => onSelect(addr)}
            className="font-mono text-xs px-3 py-1.5 rounded border transition-colors hover:border-[var(--primary)]"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
              color: "var(--muted)",
            }}
          >
            {addr.slice(0, 6)}...{addr.slice(-4)}
          </button>
        ))}
      </div>
    </div>
  );
}

function saveRecent(addr: string) {
  try {
    const prev: string[] = JSON.parse(
      localStorage.getItem("cs_recents") ?? "[]"
    );
    localStorage.setItem(
      "cs_recents",
      JSON.stringify([addr, ...prev.filter((a) => a !== addr)].slice(0, 5))
    );
  } catch {
    /* noop */
  }
}

// ── Loading Skeleton ───────────────────────────────────────────────────────

function LoadingSkeleton() {
  return (
    <div className="w-full flex flex-col gap-4 animate-pulse">
      <div
        className="h-48 rounded-lg"
        style={{ background: "var(--card)" }}
      />
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-24 rounded-lg"
            style={{ background: "var(--card)" }}
          />
        ))}
      </div>
      <div
        className="h-64 rounded-lg"
        style={{ background: "var(--card)" }}
      />
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function Home() {
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleScore(addr?: string) {
    const target = (addr ?? address).trim();
    if (!target.startsWith("0x") || target.length !== 42) {
      setError("Invalid address format. Expected 0x... (42 characters)");
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
        throw new Error(body.detail ?? `API Error: ${res.status}`);
      }
      const data: ScoreResponse = await res.json();
      setResult(data);
      saveRecent(target);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  const tier = result ? TIER_CONFIG[result.risk_tier] : null;

  return (
    <main className="flex flex-col flex-1">
      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left Panel - Input */}
        <div
          className="lg:w-96 p-6 border-b lg:border-b-0 lg:border-r flex flex-col gap-6"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex flex-col gap-1">
            <h1
              className="text-lg font-semibold"
              style={{ color: "var(--foreground)" }}
            >
              Credit Analysis
            </h1>
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Institutional-grade on-chain wallet scoring
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <label
              className="text-xs font-medium uppercase tracking-wider"
              style={{ color: "var(--muted)" }}
            >
              Wallet Address
            </label>
            <input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleScore()}
              placeholder="0x..."
              className="w-full rounded-lg border px-4 py-3 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-[var(--primary)]"
              style={{
                background: "var(--card)",
                borderColor: "var(--border)",
                color: "var(--foreground)",
              }}
            />
            <button
              onClick={() => handleScore()}
              disabled={loading}
              className="w-full px-4 py-3 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50"
              style={{
                background: "var(--primary)",
                color: "#fff",
              }}
            >
              {loading ? "ANALYZING..." : "RUN ANALYSIS"}
            </button>
          </div>

          <RecentSearches onSelect={(a) => handleScore(a)} />

          {/* Info panel */}
          <div
            className="mt-auto p-4 rounded-lg border"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
            }}
          >
            <div
              className="text-xs font-medium uppercase tracking-wider mb-2"
              style={{ color: "var(--muted)" }}
            >
              Model Info
            </div>
            <div
              className="flex flex-col gap-1 text-xs font-mono"
              style={{ color: "var(--muted)" }}
            >
              <div className="flex justify-between">
                <span>Engine</span>
                <span style={{ color: "var(--foreground)" }}>LightGBM</span>
              </div>
              <div className="flex justify-between">
                <span>Training Data</span>
                <span style={{ color: "var(--foreground)" }}>49,748 wallets</span>
              </div>
              <div className="flex justify-between">
                <span>Source</span>
                <span style={{ color: "var(--foreground)" }}>Aave V2</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="flex-1 p-6 overflow-auto">
          {!result && !loading && !error && (
            <div
              className="h-full flex flex-col items-center justify-center text-center gap-4"
              style={{ color: "var(--muted)" }}
            >
              <div
                className="w-16 h-16 rounded-full border-2 flex items-center justify-center"
                style={{ borderColor: "var(--border)" }}
              >
                <svg
                  className="w-8 h-8"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <div>
                <p className="text-sm font-medium" style={{ color: "var(--foreground)" }}>
                  No Analysis Running
                </p>
                <p className="text-xs">
                  Enter an Ethereum wallet address to begin credit assessment
                </p>
              </div>
            </div>
          )}

          {error && (
            <div
              className="p-4 rounded-lg border"
              style={{
                background: "rgba(239, 68, 68, 0.1)",
                borderColor: "rgba(239, 68, 68, 0.3)",
              }}
            >
              <p className="text-sm font-mono text-red-400">{error}</p>
            </div>
          )}

          {loading && <LoadingSkeleton />}

          {result && tier && !loading && (
            <div className="flex flex-col gap-6">
              {/* Header */}
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div className="flex flex-col gap-2">
                  <div
                    className="text-xs font-medium uppercase tracking-wider"
                    style={{ color: "var(--muted)" }}
                  >
                    Analysis Result
                  </div>
                  <div
                    className="text-sm font-mono"
                    style={{ color: "var(--foreground)" }}
                  >
                    {result.wallet_address}
                  </div>
                </div>
                <RiskBadge tier={result.risk_tier} />
              </div>

              {/* Score section */}
              <div
                className="p-6 rounded-lg border"
                style={{
                  background: "var(--card)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="text-xs font-medium uppercase tracking-wider mb-4"
                  style={{ color: "var(--muted)" }}
                >
                  Credit Score
                </div>
                <ScoreDisplay score={result.score} tier={result.risk_tier} />
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MetricCard
                  label="Probability of Default"
                  value={`${(result.probability_of_default * 100).toFixed(1)}%`}
                  highlight
                />
                <MetricCard
                  label="Credit Grade"
                  value={tier.grade}
                  subtext={tier.label}
                />
                <MetricCard
                  label="Valid Until"
                  value={result.score_valid_until}
                  subtext={result.model_version}
                />
              </div>

              {/* Risk Factors */}
              {result.top_factors.length > 0 && (
                <div
                  className="p-6 rounded-lg border"
                  style={{
                    background: "var(--card)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div
                      className="text-xs font-medium uppercase tracking-wider"
                      style={{ color: "var(--muted)" }}
                    >
                      Risk Factor Analysis
                    </div>
                    <span
                      className="text-xs font-mono px-2 py-1 rounded"
                      style={{
                        background: "var(--background)",
                        color: "var(--muted)",
                      }}
                    >
                      SHAP Values
                    </span>
                  </div>
                  <ShapChart factors={result.top_factors} />
                </div>
              )}

              {/* Disclaimer */}
              <p
                className="text-xs text-center py-4 font-mono"
                style={{ color: "var(--muted)" }}
              >
                FOR INFORMATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer
        className="border-t px-6 py-3 flex items-center justify-between text-xs font-mono"
        style={{
          borderColor: "var(--border)",
          color: "var(--muted)",
          background: "var(--background)",
        }}
      >
        <span>ChainScore Terminal v1.0</span>
        <div className="flex items-center gap-4">
          <a
            href="https://br.linkedin.com/in/andrepinheiropaes"
            className="hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            @andrepinheiropaes
          </a>
          <a
            href="https://github.com/deerws/ChainScore"
            className="hover:underline"
            target="_blank"
            rel="noreferrer"
          >
            Source Code
          </a>
        </div>
      </footer>
    </main>
  );
}
