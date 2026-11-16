// What the stage shows: a range of layers and a least size of effect. Every view and the legend follow.
import React from "react";
import { fmt, int } from "../api";

export type Filter = { lo: number; hi: number; min: number };
const layerName = (col: number) => (col < 0 ? "In" : `S${col}`);

export function Filters({ layers, withLayers, filter, onFilter, max, shown, total, noun }: {
  layers: number;
  /** probes and traces look at one layer, so only the size filter applies */
  withLayers: boolean;
  filter: Filter; onFilter: (f: Filter) => void;
  /** the largest effect among the parts, the end of the size slider */
  max: number; shown: number; total: number; noun: string;
}) {
  const active = filter.min > 0 || (withLayers && (filter.lo > -1 || filter.hi < layers - 1));
  // the size slider is quadratic, so small effects get most of its travel
  const t = max > 0 ? Math.round(100 * Math.sqrt(Math.min(1, filter.min / max))) : 0;
  return (
    <div className="ap-filters">
      {withLayers && (
        <label className="ap-filter">
          <span className="np-caption">Layers {layerName(filter.lo)} to {layerName(filter.hi)}</span>
          <span className="ap-dual">
            <input className="np-range" type="range" min={-1} max={layers - 1} step={1} value={filter.lo} aria-label="First layer"
              onChange={(e) => onFilter({ ...filter, lo: Math.min(Number(e.target.value), filter.hi) })} />
            <input className="np-range" type="range" min={-1} max={layers - 1} step={1} value={filter.hi} aria-label="Last layer"
              onChange={(e) => onFilter({ ...filter, hi: Math.max(Number(e.target.value), filter.lo) })} />
          </span>
        </label>
      )}
      <label className="ap-filter">
        <span className="np-caption">Effect of at least {fmt(filter.min, 3).replace("+", "")}</span>
        <input className="np-range" type="range" min={0} max={100} step={1} value={t} aria-label="Least effect"
          onChange={(e) => onFilter({ ...filter, min: max * (Number(e.target.value) / 100) ** 2 })} />
      </label>
      <span className="ap-filter-note">
        {active ? `${int(shown)} of ${int(total)} ${noun} shown; ${int(total - shown)} are left out by the filter. ` : `All ${int(total)} ${noun} shown. `}
        {active && <button className="np-link" onClick={() => onFilter({ lo: -1, hi: layers - 1, min: 0 })}>Reset</button>}
      </span>
    </div>
  );
}
