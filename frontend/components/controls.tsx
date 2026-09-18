"use client";

import type { HyperParam } from "@/lib/types";

export function ParamControl({
  p,
  value,
  onChange,
}: {
  p: HyperParam;
  value: number | boolean | string;
  onChange: (v: number | boolean | string) => void;
}) {
  const label = (
    <label>
      {p.label}
      {p.unit ? <small> ({p.unit})</small> : null}
    </label>
  );
  const tip = p.tooltip ? (
    <span className="tooltip">
      <span className="info">?</span>
      <span className="dttip">
        <b>{p.label}</b>
        <div>{p.tooltip}</div>
        {p.hint ? <div className="hint">💡 {p.hint}</div> : null}
      </span>
    </span>
  ) : null;

  if (p.type === "bool") {
    return (
      <div className="field">
        <div className="row">
          <span
            style={{ cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "4px" }}
            onClick={() => onChange(!Boolean(value))}
          >
            <span>
              {p.label}
              {p.unit ? <small> ({p.unit})</small> : null}
            </span>
            <span onClick={(e) => e.stopPropagation()}>{tip}</span>
          </span>
          <label className="switch">
            <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
            <span className="track" />
          </label>
        </div>
      </div>
    );
  }

  if (p.type === "select") {
    return (
      <div className="field">
        <div className="row">
          <span>
            {label} {tip}
          </span>
        </div>
        <select value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {Object.entries(p.options || {}).map(([k, v]) => (
            <option key={k} value={k}>
              {v}
            </option>
          ))}
        </select>
      </div>
    );
  }

  const num = Number(value);
  return (
    <div className="field">
      <div className="row">
        <span>
          {label} {tip}
        </span>
        <span className="val mono">{num.toFixed(p.type === "int" ? 0 : 2)}</span>
      </div>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={num}
        onChange={(e) => onChange(p.type === "int" ? Math.round(Number(e.target.value)) : Number(e.target.value))}
      />
    </div>
  );
}

export function SliderRow({
  label,
  min,
  max,
  step,
  value,
  onChange,
  unit = "",
  hint,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  unit?: string;
  hint?: string;
}) {
  return (
    <div className="field">
      <div className="row">
        <span>
          {label} {hint ? <TooltipIcon label={label} text={hint} /> : null}
        </span>
        <span className="val mono">
          {value}
          {unit && ` ${unit}`}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function BoolRow({
  label,
  text,
  value,
  onChange,
}: {
  label: string;
  text: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="field">
      <div className="row">
        <span
          style={{ cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "4px" }}
          onClick={() => onChange(!value)}
        >
          <span>{label}</span>
          <span onClick={(e) => e.stopPropagation()}>
            <TooltipIcon label={label} text={text} />
          </span>
        </span>
        <label className="switch">
          <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
          <span className="track" />
        </label>
      </div>
    </div>
  );
}

export function TooltipIcon({ label, text, hint }: { label: string; text: string; hint?: string }) {
  return (
    <span className="tooltip">
      <span className="info">?</span>
      <span className="dttip">
        <b>{label}</b>
        <div>{text}</div>
        {hint ? <div className="hint">💡 {hint}</div> : null}
      </span>
    </span>
  );
}