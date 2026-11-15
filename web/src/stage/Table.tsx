// Every measured part as a row, sortable by any column.
import React, { useMemo, useRef, useState } from "react";
import { Detail, fmt, int } from "../api";
import { Part, familyColor, familyOf } from "../scene";

type Row = { part: Detail; parent?: Part };
type Col = { id: string; label: string; num: boolean; get: (r: Row) => number | string | null; show: (r: Row) => string };
const num = (v: number | null | undefined) => (v == null ? null : v);
const COLS: Col[] = [
  { id: "label", label: "Part", num: false, get: (r) => r.part.label, show: (r) => r.part.label },
  { id: "of", label: "Of", num: false, get: (r) => r.parent?.label ?? "", show: (r) => r.parent?.label ?? "" },
  { id: "kind", label: "Type", num: false, get: (r) => r.part.kind, show: (r) => r.part.kind },
  { id: "layer", label: "Layer", num: true, get: (r) => num(r.part.layer ?? r.parent?.layer), show: (r) => String(r.part.layer ?? r.parent?.layer ?? "—") },
  { id: "position", label: "Position", num: true, get: (r) => num(r.part.position), show: (r) => (r.part.position == null ? "—" : String(r.part.position)) },
  { id: "direct", label: "Direct", num: true, get: (r) => num(r.part.direct ?? r.part.activation_contribution), show: (r) => fmt(r.part.direct ?? r.part.activation_contribution) },
  { id: "total", label: "Total", num: true, get: (r) => num(r.part.total), show: (r) => fmt(r.part.total) },
  { id: "indirect", label: "Indirect", num: true, get: (r) => (r.part.total != null && r.part.direct != null ? r.part.total - r.part.direct : null), show: (r) => (r.part.total != null && r.part.direct != null ? fmt(r.part.total - r.part.direct) : "—") },
  { id: "kl", label: "KL", num: true, get: (r) => num(r.part.kl), show: (r) => (r.part.kl == null ? "—" : r.part.kl.toExponential(2)) },
];

export function Table({ rows, selected, onSelect, brush, onBrush, other }: {
  rows: Row[]; selected: string; onSelect: (id: string) => void;
  /** rows marked together, with Ctrl or Shift */
  brush: Set<string> | null; onBrush: (ids: string[]) => void;
  /** the same part's value in the run compared with */
  other?: Map<string, number | null>;
}) {
  const [sort, setSort] = useState<{ id: string; down: boolean } | null>(null);
  const last = useRef(-1);
  const all = useMemo<Col[]>(() => (other ? [
    ...COLS,
    { id: "other", label: "Other run", num: true, get: (r) => num(other.get(r.part.id)), show: (r) => fmt(other.get(r.part.id)) },
  ] : COLS), [other]);
  // columns no row has a value for are left out
  const cols = useMemo(() => all.filter((c) => c.id === "label" || rows.some((r) => { const v = c.get(r); return v != null && v !== ""; })), [rows, all]);
  const sorted = useMemo(() => {
    const col = sort && all.find((c) => c.id === sort.id);
    if (!sort || !col) return rows;
    const key = (r: Row) => { const v = col.get(r); return typeof v === "number" ? Math.abs(v) : v; };
    return [...rows].sort((a, b) => {
      const x = key(a), y = key(b);
      if (x == null || x === "") return 1; // rows without a value stay at the end
      if (y == null || y === "") return -1;
      const d = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true });
      return sort.down ? -d : d;
    });
  }, [rows, sort, all]);
  const click = (e: React.MouseEvent, id: string, i: number) => {
    if (e.shiftKey && last.current >= 0) {
      const [a, b] = [Math.min(last.current, i), Math.max(last.current, i)];
      onBrush([...new Set([...(brush || []), ...sorted.slice(a, b + 1).map((r) => r.part.id)])]);
    } else if (e.ctrlKey || e.metaKey) {
      const next = new Set(brush || []);
      if (next.has(id)) next.delete(id); else next.add(id);
      onBrush([...next]);
    } else onSelect(id === selected ? "" : id);
    last.current = i;
  };
  return (
    <div className="ap-table-wrap">
      <table className="np-table ap-table">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.id} className={c.num ? "np-num" : ""}>
                <button className={sort?.id === c.id ? "is-sorted" : ""} title={c.num ? "Sort by size" : "Sort"}
                  onClick={() => setSort((s) => (s?.id === c.id ? (s.down === c.num ? { id: c.id, down: !c.num } : null) : { id: c.id, down: c.num }))}>
                  {c.label}{sort?.id === c.id ? (sort.down ? " ↓" : " ↑") : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.part.id} className={`${r.part.id === selected ? "is-active" : ""} ${brush?.has(r.part.id) ? "is-brushed" : ""}`} onClick={(e) => click(e, r.part.id, i)}>
              {cols.map((c) => (
                <td key={c.id} className={c.num ? "np-num" : c.id === "label" ? "" : "ap-quiet"}>
                  {c.id === "label" && <span className="np-dot" style={{ color: familyColor(familyOf(r.part)) }} />}{c.id === "label" ? " " : ""}{c.show(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="np-note ap-drawn">{int(rows.length)} parts. Numeric columns sort by size, without the sign. Shift or Ctrl marks several rows; they light up in the other views.</p>
    </div>
  );
}
