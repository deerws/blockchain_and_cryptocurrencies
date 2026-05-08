import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Nav from "./components/Nav";
import "./globals.css";

const geistSans = Geist({ subsets: ["latin"] });
const geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ChainScore | On-Chain Credit Intelligence",
  description:
    "Institutional-grade credit scoring for Ethereum wallets. Real-time risk assessment powered by on-chain behavioral analysis.",
};

export const viewport: Viewport = {
  themeColor: "#0a0a0b",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.className} ${geistMono.className} bg-background`} suppressHydrationWarning>
      <body className="min-h-screen flex flex-col antialiased">
        <Nav />
        {children}
      </body>
    </html>
  );
}
