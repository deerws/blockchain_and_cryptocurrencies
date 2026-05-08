import type { Metadata, Viewport } from "next";
import { Inter, Newsreader, IBM_Plex_Mono } from "next/font/google";
import Nav from "./components/Nav";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter"
});

const newsreader = Newsreader({ 
  subsets: ["latin"],
  variable: "--font-newsreader",
  style: ["normal", "italic"]
});

const ibmPlexMono = IBM_Plex_Mono({ 
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-ibm-mono"
});

export const metadata: Metadata = {
  title: "ChainScore | Wallet Credit Intelligence",
  description:
    "Institutional-grade on-chain credit risk analysis and behavioral scoring for Ethereum wallets.",
};

export const viewport: Viewport = {
  themeColor: "#FCFCFB",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html 
      lang="en" 
      className={`${inter.variable} ${newsreader.variable} ${ibmPlexMono.variable} bg-background`} 
      suppressHydrationWarning
    >
      <body className="min-h-screen flex flex-col antialiased font-sans">
        <Nav />
        {children}
      </body>
    </html>
  );
}
