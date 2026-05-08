"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Nav() {
  const path = usePathname();

  const links = [
    { href: "/", label: "Terminal" },
    { href: "/about", label: "Methodology" },
  ];

  return (
    <header
      className="border-b px-4 sm:px-6 h-14 flex items-center gap-6 sticky top-0 z-50"
      style={{
        background: "var(--background)",
        borderColor: "var(--border)",
      }}
    >
      {/* Logo */}
      <Link href="/" className="flex items-center gap-3 shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--positive)] animate-pulse" />
          <span
            className="font-semibold text-sm tracking-tight"
            style={{ color: "var(--foreground)" }}
          >
            CHAINSCORE
          </span>
        </div>
        <span
          className="text-xs px-2 py-0.5 rounded font-mono"
          style={{
            background: "var(--card)",
            color: "var(--muted)",
            border: "1px solid var(--border)",
          }}
        >
          v1.0
        </span>
      </Link>

      {/* Divider */}
      <div className="w-px h-5" style={{ background: "var(--border)" }} />

      {/* Nav links */}
      <nav className="flex items-center gap-1">
        {links.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className="px-3 py-1.5 rounded text-xs font-medium transition-colors"
            style={{
              color: path === href ? "var(--foreground)" : "var(--muted)",
              background: path === href ? "var(--card)" : "transparent",
            }}
          >
            {label}
          </Link>
        ))}
      </nav>

      {/* Right side - Status indicators */}
      <div className="ml-auto flex items-center gap-4">
        <div className="hidden sm:flex items-center gap-4 text-xs font-mono" style={{ color: "var(--muted)" }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--positive)]" />
            <span>API Online</span>
          </div>
          <div className="flex items-center gap-2">
            <span>ETH Mainnet</span>
          </div>
        </div>
        <a
          href="https://github.com/deerws/ChainScore"
          target="_blank"
          rel="noreferrer"
          className="text-xs font-medium px-3 py-1.5 rounded transition-colors"
          style={{
            color: "var(--muted)",
            background: "var(--card)",
            border: "1px solid var(--border)",
          }}
        >
          GitHub
        </a>
      </div>
    </header>
  );
}
