"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, useCallback } from "react";

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

export default function Nav() {
  const { dark, toggle } = useTheme();
  const path = usePathname();

  const links = [
    { href: "/",       label: "Score" },
    { href: "/about",  label: "About" },
  ];

  return (
    <header
      className="border-b px-4 sm:px-6 py-3 flex items-center gap-3 sticky top-0 z-10 backdrop-blur-sm"
      style={{
        background: "color-mix(in srgb, var(--card) 90%, transparent)",
        borderColor: "var(--border)",
      }}
    >
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2.5 shrink-0">
        <div
          className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center"
          style={{ background: "#185FA5" }}
        >
          <span className="text-white font-bold text-sm">C</span>
        </div>
        <span className="font-semibold text-base sm:text-lg" style={{ color: "var(--fg)" }}>
          ChainScore
        </span>
      </Link>

      {/* Nav links */}
      <nav className="flex items-center gap-1 ml-4">
        {links.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className="px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
            style={{
              color: path === href ? "var(--fg)" : "var(--muted)",
              background: path === href ? "var(--border)" : "transparent",
            }}
          >
            {label}
          </Link>
        ))}
      </nav>

      {/* Right side */}
      <div className="ml-auto flex items-center gap-2 sm:gap-3">
        <a
          href="https://github.com/deerws/ChainScore"
          target="_blank"
          rel="noreferrer"
          className="hidden sm:block text-xs font-medium transition-opacity hover:opacity-70"
          style={{ color: "var(--muted)" }}
        >
          GitHub ↗
        </a>
        <button
          onClick={toggle}
          className="w-8 h-8 sm:w-9 sm:h-9 rounded-lg border flex items-center justify-center transition-opacity hover:opacity-70"
          style={{ borderColor: "var(--border)", background: "var(--bg)", color: "var(--muted)" }}
          aria-label="Toggle dark mode"
        >
          {dark ? "☀️" : "🌙"}
        </button>
      </div>
    </header>
  );
}
