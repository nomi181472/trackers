import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tracker Failure Simulator",
  description: "See why object trackers fail — profile ByteTrack, BoT-SORT, OC-SORT and friends on scenes designed to break them.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}