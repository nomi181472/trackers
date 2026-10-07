import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multi-Object Tracker Simulator | Standard & Production",
  description: "Profile and stress-test computer vision trackers (ByteTrack, BoT-SORT, OC-SORT) with Standard failure scenarios and Production-grade SLAs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}