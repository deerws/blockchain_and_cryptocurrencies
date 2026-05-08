import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Nav from "./components/Nav";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter"
});

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ["latin"],
  variable: "--font-jetbrains"
});

export const metadata: Metadata = {
  title: "ChainScore | Wallet Credit Intelligence",
  description:
    "Institutional-grade on-chain credit risk analysis and behavioral scoring for Ethereum wallets.",
};

export const viewport: Viewport = {
  themeColor: "#FAFAF9",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html 
      lang="en" 
      className={`${inter.variable} ${jetbrainsMono.variable} bg-[var(--background)]`} 
      suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                const stored = localStorage.getItem('chainscore-theme');
                if (stored === 'dark' || (!stored && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body className="min-h-screen flex flex-col antialiased font-sans" style={{ background: 'var(--background)', color: 'var(--foreground)' }}>
        <Nav />
        {children}
      </body>
    </html>
  );
}
