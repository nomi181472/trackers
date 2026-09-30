"use client";

import { useState } from "react";
import type { Catalog, ParamValues } from "@/lib/types";
import { ParamControl } from "@/components/controls";

const BADGE: Record<string, string> = {
  ultralytics: "badge-ul",
  custom: "badge-cu",
  opencv: "badge-cv",
};

export function TrackerPicker({
  catalog,
  selected,
  setSelected,
}: {
  catalog: Catalog;
  selected: Record<string, ParamValues | null>;
  setSelected: (s: Record<string, ParamValues | null>) => void;
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => {
    setSelected({ ...selected, [id]: selected[id] ? null : { ...(catalog.defaults[id] || {}) } });
  };

  const good = catalog.trackers.filter((t) => t.available);
  const bad = catalog.trackers.filter((t) => !t.available);

  return (
    <div>
      {good.map((t) => (
        <TrackerCard
          key={t.id}
          t={t}
          enabled={Boolean(selected[t.id])}
          open={Boolean(open[t.id])}
          onToggle={() => toggle(t.id)}
          onOpen={() => setOpen((o) => ({ ...o, [t.id]: !o[t.id] }))}
          params={selected[t.id] || catalog.defaults[t.id] || {}}
          setParams={(p) => setSelected({ ...selected, [t.id]: p })}
        />
      ))}
      {bad.length ? (
        <div className="collapse-head" onClick={() => setOpen((o) => ({ ...o, _na: !o._na }))}>
          <span style={{ transform: `rotate(${open._na ? 90 : 0}deg)`, display: "inline-block" }}>▸</span>
          Not available in this build ({bad.length})
        </div>
      ) : null}
      {open._na
        ? bad.map((t) => (
            <div key={t.id} className="tr-card" style={{ opacity: 0.6 }}>
              <div className="tr-head">
                <input type="checkbox" className="chk" disabled />
                <span className="nm">{t.name}</span>
                <span className={`badge badge-na`}>not available</span>
              </div>
            </div>
          ))
        : null}
    </div>
  );
}

export function TrackerCard({
  t,
  enabled,
  open,
  onToggle,
  onOpen,
  params,
  setParams,
}: {
  t: Catalog["trackers"][number];
  enabled: boolean;
  open: boolean;
  onToggle: () => void;
  onOpen: () => void;
  params: ParamValues;
  setParams: (p: ParamValues) => void;
}) {
  return (
    <div className={`tr-card ${open ? "open" : ""}`}>
      <div className="tr-head" onClick={onOpen}>
        <input
          type="checkbox"
          className="chk"
          checked={enabled}
          onClick={(e) => e.stopPropagation()}
          onChange={onToggle}
        />
        <span className="nm">{t.name}</span>
        <span className="tag">{t.tagline}</span>
        <span className={`badge ${BADGE[t.engine] ?? "badge-cu"}`}>{t.engine}</span>
        {t.mode === "single" ? <span className="badge badge-single">single</span> : null}
        <span className="chev">▸</span>
      </div>
      {open ? (
        <div className="tr-body">
          <small>{t.description}</small>
          <div className="metrics" style={{ padding: "8px 0" }}>
            {t.failure_modes.map((f) => (
              <span key={f} className="metric">⚠️ {f}</span>
            ))}
          </div>
          {t.params.map((p) => (
            <ParamControl key={p.key} p={p} value={params[p.key] ?? p.default} onChange={(v) => setParams({ ...params, [p.key]: v })} />
          ))}
          {t.mode === "single" ? (
            <small>
              Single-object tracker: it will be initialised on the first ground-truth box, and switched to a new object whenever the previous one
              is lost.
            </small>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}