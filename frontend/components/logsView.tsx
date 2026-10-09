"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { LogFileInfo, LogPage } from "@/lib/types";
import { getLogFiles, getLogLines } from "@/lib/api";

export function LogsView({ active = false }: { active?: boolean }) {
  const [selectedWorker, setSelectedWorker] = useState<string>("worker-n");
  const [files, setFiles] = useState<LogFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [lines, setLines] = useState<string[]>([]);
  const [nextCursor, setNextCursor] = useState<number | null>(null);
  const [totalLines, setTotalLines] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMore, setLoadingMore] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [filterQuery, setFilterQuery] = useState<string>("");

  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomAnchorRef = useRef<HTMLDivElement>(null);
  const prevScrollHeightRef = useRef<number>(0);

  // Load available files only when active or worker changes
  const fetchFiles = useCallback(async () => {
    if (!active) return;
    try {
      setError(null);
      const res = await getLogFiles(selectedWorker);
      setFiles(res.files);
      if (res.files.length > 0) {
        setSelectedFile(res.files[0].filename);
      } else {
        setSelectedFile("");
        setLines([]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setFiles([]);
      setSelectedFile("");
      setLines([]);
    }
  }, [active, selectedWorker]);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  // Load latest chunk when selected file changes, but only if tab is active
  const loadLatestLogs = useCallback(async (filename: string) => {
    if (!active || !filename) return;
    setLoading(true);
    setError(null);
    try {
      const data: LogPage = await getLogLines({ file: filename, limit: 100, worker: selectedWorker });
      setLines(data.lines);
      setNextCursor(data.next_cursor);
      setTotalLines(data.total_lines);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [active, selectedWorker]);

  useEffect(() => {
    if (active && selectedFile) {
      loadLatestLogs(selectedFile);
    }
  }, [active, selectedFile, loadLatestLogs]);

  // Periodic polling only if active and autoRefresh is explicitly enabled
  useEffect(() => {
    if (!active || !autoRefresh || !selectedFile) return;
    const interval = setInterval(async () => {
      try {
        const data: LogPage = await getLogLines({ file: selectedFile, limit: 100, worker: selectedWorker });
        setTotalLines(data.total_lines);
        setLines((prev) => {
          if (data.lines.length === 0) return prev;
          if (prev.length <= 100) {
            return data.lines;
          }
          return prev;
        });
      } catch {
        // silent background refresh error
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [active, autoRefresh, selectedFile, selectedWorker]);

  // Load older lines when user scrolls to top/bottom
  const loadOlderLogs = async () => {
    if (nextCursor === null || loadingMore || !selectedFile) return;
    setLoadingMore(true);

    if (scrollRef.current) {
      prevScrollHeightRef.current = scrollRef.current.scrollHeight;
    }

    try {
      const data: LogPage = await getLogLines({
        file: selectedFile,
        cursor: nextCursor,
        limit: 100,
        worker: selectedWorker,
      });

      setLines((prev) => [...data.lines, ...prev]);
      setNextCursor(data.next_cursor);
      setTotalLines(data.total_lines);

      // Preserve scroll position
      requestAnimationFrame(() => {
        if (scrollRef.current) {
          const newScrollHeight = scrollRef.current.scrollHeight;
          const diff = newScrollHeight - prevScrollHeightRef.current;
          scrollRef.current.scrollTop += diff;
        }
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingMore(false);
    }
  };

  // Scroll listener for infinite scroll towards older lines
  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop } = scrollRef.current;
    // When user scrolls near top of container, fetch older records
    if (scrollTop < 40 && nextCursor !== null && !loadingMore) {
      loadOlderLogs();
    }
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  const filteredLines = filterQuery
    ? lines.filter((l) => l.toLowerCase().includes(filterQuery.toLowerCase()))
    : lines;

  return (
    <div className="panel" style={{ marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <h3 style={{ margin: 0 }}>📋 Application Logs</h3>
          <span style={{ fontSize: 12, color: "var(--muted)" }}>
            ({totalLines} total lines in file)
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {/* Worker Selector Dropdown */}
          <div style={{ minWidth: 130 }}>
            <select
              value={selectedWorker}
              onChange={(e) => setSelectedWorker(e.target.value)}
              disabled={loading}
              style={{
                padding: "6px 10px",
                fontSize: 12.5,
                fontWeight: 600,
                borderColor: "var(--accent)",
                color: "var(--text)",
                background: "var(--bg-2)",
              }}
              title="Select which worker node to inspect logs from"
            >
              <option value="worker-n">⚡ worker-n (Active)</option>
            </select>
          </div>

          {/* Daily log file dropdown */}
          <div style={{ minWidth: 200 }}>
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              disabled={loading}
              style={{ padding: "6px 10px", fontSize: 12.5 }}
            >
              {files.length === 0 ? (
                <option value="">No log files found</option>
              ) : (
                files.map((f) => (
                  <option key={f.filename} value={f.filename}>
                    📅 {f.date} ({Math.round(f.size_bytes / 1024)} KB)
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Quick search/filter input */}
          <input
            type="text"
            placeholder="Filter logs (e.g. ERROR, CRASH)..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            style={{
              width: 200,
              background: "var(--bg-3)",
              border: "1px solid var(--line)",
              color: "var(--text)",
              borderRadius: 6,
              padding: "6px 10px",
              fontSize: 12,
            }}
          />

          <button
            type="button"
            className="btn ghost"
            style={{ padding: "6px 12px", fontSize: 12 }}
            onClick={() => {
              fetchFiles();
              if (selectedFile) loadLatestLogs(selectedFile);
            }}
            disabled={loading}
          >
            🔄 Refresh
          </button>

          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--muted)", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              style={{ accentColor: "var(--accent)" }}
            />
            Auto-tail (4s)
          </label>
        </div>
      </div>

      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
        <span>Showing {filteredLines.length} lines. {nextCursor !== null ? `Scroll up or click 'Load Older' to retrieve earlier history.` : `At oldest recorded entry for this date.`}</span>
        {nextCursor !== null && (
          <button
            type="button"
            onClick={loadOlderLogs}
            disabled={loadingMore}
            className="btn ghost"
            style={{ padding: "3px 8px", fontSize: 11 }}
          >
            {loadingMore ? "Loading older lines..." : `⬆️ Load older (${nextCursor} remaining above)`}
          </button>
        )}
      </div>

      {error && <div className="err" style={{ marginBottom: 10 }}>{error}</div>}

      {/* Terminal log viewer */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        style={{
          background: "#080b10",
          border: "1px solid var(--line)",
          borderRadius: 8,
          padding: "12px",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 12,
          lineHeight: 1.5,
          color: "#d8dee9",
          maxHeight: "560px",
          minHeight: "300px",
          overflowY: "auto",
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
        }}
      >
        {loading && lines.length === 0 ? (
          <div style={{ color: "var(--muted)", padding: 20, textAlign: "center" }}>
            <span className="spin" /> Loading log lines...
          </div>
        ) : filteredLines.length === 0 ? (
          <div style={{ color: "var(--muted)", padding: 20, textAlign: "center" }}>
            No log entries found for this file.
          </div>
        ) : (
          filteredLines.map((line, idx) => {
            const isError = line.includes("[ERROR]") || line.includes("[CRITICAL]") || line.includes("Traceback");
            const isWarn = line.includes("[WARNING]");
            const isJob = line.includes("[trace:job-");
            return (
              <div
                key={idx}
                style={{
                  color: isError ? "#ff7b88" : isWarn ? "#ffd166" : isJob ? "#72d1ff" : "#d8dee9",
                  backgroundColor: isError ? "rgba(255, 93, 108, 0.08)" : "transparent",
                  padding: isError ? "2px 4px" : "1px 0",
                  borderRadius: 3,
                }}
              >
                {line}
              </div>
            );
          })
        )}
        <div ref={bottomAnchorRef} />
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
        <button
          type="button"
          onClick={scrollToBottom}
          className="btn ghost"
          style={{ padding: "4px 10px", fontSize: 11.5 }}
        >
          ⬇️ Jump to Bottom
        </button>
      </div>
    </div>
  );
}
