"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { label: "Dashboard", href: "/" },
  { label: "Watchlist", href: "/watchlist" },
  { label: "Alerts", href: "/alerts" },
  { label: "Reports", href: "/reports" },
  { label: "API", href: "/api-docs" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 bg-background border-b border-border">
      <div className="max-w-[1400px] mx-auto px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <span className="text-lg font-medium tracking-tight text-foreground">
              Chain<span className="text-primary">Score</span>
            </span>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 py-1.5 text-[13px] font-medium transition-colors rounded ${
                    isActive
                      ? "text-foreground bg-accent"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {/* Wallet Identity Block */}
          <div className="flex items-center gap-4">
            <div className="hidden lg:flex items-center gap-3 text-right">
              <div>
                <p className="text-[11px] uppercase tracking-wider text-muted-foreground">
                  Analyzed 2 min ago
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2 pl-4 border-l border-border">
              {/* Avatar placeholder */}
              <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                <span className="text-xs font-medium text-primary">VB</span>
              </div>
              <div className="hidden sm:block">
                <p className="text-sm font-medium text-foreground">vitalik.eth</p>
                <p className="text-[11px] text-muted-foreground font-mono">0xd8dA...6045</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
