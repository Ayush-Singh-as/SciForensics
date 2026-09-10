import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SciForensics — Image Integrity Analysis",
  description:
    "Forensic detection of image reuse and manipulation in scientific figures: " +
    "embedding similarity, local feature matching and geometric verification.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/* Fonts are loaded rather than assumed: the whole interface depends on
          tabular numerals, and a fallback without them makes live readouts
          jitter as digits change width. */}
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
