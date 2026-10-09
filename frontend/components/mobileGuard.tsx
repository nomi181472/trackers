"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "tracker_mobile_override";

export function MobileGuard({ children }: { children: React.ReactNode }) {
  const [isMobile, setIsMobile] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Check if user previously chose "Run Anyway" this session
    const override = sessionStorage.getItem(STORAGE_KEY);
    if (override === "1") {
      setDismissed(true);
      return;
    }
    // Detect mobile: screen width < 768px or touch-primary device
    const checkMobile = () => {
      const narrow = window.innerWidth < 768;
      const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
      setIsMobile(narrow || coarsePointer);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const handleRunAnyway = () => {
    sessionStorage.setItem(STORAGE_KEY, "1");
    setDismissed(true);
  };

  // Don't render anything until mounted (avoid SSR mismatch)
  if (!mounted) return <>{children}</>;

  if (isMobile && !dismissed) {
    return (
      <div className="mobile-guard-overlay">
        <div className="mobile-guard-card">
          <div className="mobile-guard-icon">🖥️</div>
          <h2 className="mobile-guard-title">Desktop Required</h2>
          <p className="mobile-guard-desc">
            <strong>Tracker Lab</strong> is a complex multi-panel diagnostic
            tool designed for desktop and laptop screens. It requires a
            viewport of at least <strong>768 px</strong> wide to display
            correctly — charts, side-by-side tracker metrics, and video
            previews won&apos;t fit on this screen.
          </p>
          <p className="mobile-guard-hint">
            📌 Please open this URL on a <strong>desktop or laptop</strong>{" "}
            browser for the best experience.
          </p>
          <div className="mobile-guard-actions">
            <a
              href={typeof window !== "undefined" ? window.location.href : "#"}
              className="mobile-guard-btn-primary"
            >
              📋 Copy Link
            </a>
            <button
              type="button"
              className="mobile-guard-btn-ghost"
              onClick={handleRunAnyway}
            >
              ⚠️ Run Anyway (Unsupported)
            </button>
          </div>
          <p className="mobile-guard-warn">
            Continuing on mobile may result in broken layouts and degraded
            functionality.
          </p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
