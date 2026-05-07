"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "";

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

const TIER_CONFIG: Record<
  RiskTier,
  { label: string; color: string; bg: string }
> = {
  very_low:  { label: "Very Low Risk",  color: "text-emerald-700", bg: "bg-emerald-50 border-emerald-200" },
  low:       { label: "Low Risk",       color: "text-green-700",   bg: "bg-green-50 border-green-200"     },
  medium:    { label: "Medium Risk",    color: "text-yellow-700",  bg: "bg-yellow-50 border-yellow-200"   },
  high:      { label: "High Risk",      color: "text-orange-700",  bg: "bg-orange-50 border-orange-200"   },
  very_high: { label: "Very High Risk", color: "text-red-700",     bg: "bg-red-50 border-red-200"         },
};

function ScoreGauge({ score }: { score: number }) {
  const pct = (score / 1000) * 100;
  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-7xl font-bold tabular-nums text-gray-900">{score}</span>
      <span className="text-sm text-gray-500">out of 1000</span>
      <div className="w-full bg-gray-200 rounded-full h-3 mt-1">
        <div
          className="h-3 rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #ef4444 0%, #f59e0b 40%, #10b981 80%)",
          }}
        />
      </div>
      <div className="flex justify-between w-full text-xs text-gray-400">
        <span>0 — Highest Risk</span>
        <span>1000 — Lowest Risk</span>
      </div>
    </div>
  );
}

function FactorBar({ factor }: { factor: ShapFactor }) {
  const isRisk = factor.direction === "increases_risk";
  const label = factor.feature.replace(/_/g, " ");
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-44 truncate text-gray-600 capitalize">{label}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${isRisk ? "bg-red-400" : "bg-emerald-400"}`}
          style={{ width: `${Math.min(100, Math.abs(factor.shap_value) * 20)}%` }}
        />
      </div>
      <span className={`w-32 text-right text-xs font-medium ${isRisk ? "text-red-600" : "text-emerald-600"}`}>
        {isRisk ? "↑ increases risk" : "↓ decreases risk"}
      </span>
    </div>
  );
}

export default function Home() {
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleScore() {
    const addr = address.trim();
    if (!addr.startsWith("0x") || addr.length !== 42) {
      setError("Enter a valid Ethereum address (0x… 42 chars).");
      return;
    }
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
        body: JSON.stringify({ wallet_address: addr, include_shap: true }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `API error ${res.status}`);
      }
      setResult(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error.");
    } finally {
      setLoading(false);
    }
  }

  const tier = result ? TIER_CONFIG[result.risk_tier] : null;

  return (
    <main className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white px-6 py-4 flex items-center gap-3">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "#185FA5" }}
        >
          <span className="text-white font-bold text-sm">C</span>
        </div>
        <span className="font-semibold text-gray-900 text-lg">ChainScore</span>
        <span className="ml-auto text-xs text-gray-400">
          On-chain credit scoring · Ethereum Mainnet
        </span>
      </header>

      <div className="flex-1 flex flex-col items-center justify-start px-4 py-16 gap-10">
        {/* Hero */}
        <div className="text-center max-w-xl">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Credit score any Ethereum wallet
          </h1>
          <p className="text-gray-500 text-base">
            Powered by Aave V2 liquidation history and on-chain behavioral
            features. Enter any wallet address to get a 0–1000 credit score and
            probability of default.
          </p>
        </div>

        {/* Search */}
        <div className="w-full max-w-2xl flex gap-2">
          <input
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleScore()}
            placeholder="0x… Ethereum wallet address"
            className="flex-1 rounded-xl border border-gray-300 px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
          />
          <button
            onClick={handleScore}
            disabled={loading}
            className="px-6 py-3 rounded-xl text-white font-semibold text-sm disabled:opacity-60 transition-opacity"
            style={{ background: "#185FA5" }}
          >
            {loading ? "Scoring…" : "Score"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="w-full max-w-2xl rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* Result */}
        {result && tier && (
          <div className="w-full max-w-2xl flex flex-col gap-4">
            <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-6 flex flex-col gap-6">
              <div className="flex items-center justify-between gap-4">
                <span className="text-sm text-gray-500 font-mono truncate">
                  {result.wallet_address}
                </span>
                <span
                  className={`shrink-0 px-3 py-1 rounded-full text-xs font-semibold border ${tier.color} ${tier.bg}`}
                >
                  {tier.label}
                </span>
              </div>

              <ScoreGauge score={result.score} />

              <div className="grid grid-cols-3 gap-4 pt-2 border-t border-gray-100">
                <div className="flex flex-col items-center gap-1">
                  <span className="text-xs text-gray-400">Prob. of Default</span>
                  <span className="text-lg font-bold text-gray-900">
                    {(result.probability_of_default * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span className="text-xs text-gray-400">Model</span>
                  <span className="text-sm font-semibold text-gray-700">
                    {result.model_version}
                  </span>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <span className="text-xs text-gray-400">Valid until</span>
                  <span className="text-sm font-semibold text-gray-700">
                    {result.score_valid_until}
                  </span>
                </div>
              </div>
            </div>

            {result.top_factors.length > 0 && (
              <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-6 flex flex-col gap-4">
                <h2 className="text-sm font-semibold text-gray-700">
                  Top factors influencing this score
                </h2>
                <div className="flex flex-col gap-3">
                  {result.top_factors.map((f) => (
                    <FactorBar key={f.feature} factor={f} />
                  ))}
                </div>
                <p className="text-xs text-gray-400">
                  Computed via SHAP (SHapley Additive exPlanations) on the
                  LightGBM model.
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-200 px-6 py-4 text-center text-xs text-gray-400">
        ChainScore · Built by{" "}
        <a
          href="https://br.linkedin.com/in/andrepinheiropaes"
          className="underline hover:text-gray-600"
          target="_blank"
          rel="noreferrer"
        >
          André Pinheiro Paes
        </a>{" "}
        · For educational and research purposes only
      </footer>
    </main>
  );
}
