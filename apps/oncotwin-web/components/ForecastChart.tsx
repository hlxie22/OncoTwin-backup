'use client';

import type { Forecast } from '@/lib/types';

const WIDTH = 760;
const HEIGHT = 310;
const PAD_L = 54;
const PAD_R = 24;
const PAD_T = 24;
const PAD_B = 42;

function pathFor(values: number[] | undefined | null) {
  if (!values?.length) return '';
  return values.map((raw, i) => {
    const v = Math.max(0, Math.min(1, raw));
    const x = PAD_L + (i / Math.max(values.length - 1, 1)) * (WIDTH - PAD_L - PAD_R);
    const y = PAD_T + (1 - v) * (HEIGHT - PAD_T - PAD_B);
    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

export function ForecastChart({ forecast, showPost = false }: { forecast: Forecast; showPost?: boolean }) {
  const curves = forecast.curves;
  if (!curves) return null;
  const pre = pathFor(curves.pre_pfs);
  const selected = pathFor(curves.selected_pfs);
  const post = pathFor(curves.post_pfs);
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks = [0, 6, 12, 18, 24];

  return (
    <div className="trajectory-chart-wrap">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="trajectory-chart" role="img" aria-label="Progression-free outlook over 24 months, before and after the newest scan">
        {yTicks.map((v) => {
          const y = PAD_T + (1 - v) * (HEIGHT - PAD_T - PAD_B);
          return (
            <g key={v}>
              <line x1={PAD_L} x2={WIDTH - PAD_R} y1={y} y2={y} className="chart-grid" />
              <text x={PAD_L - 12} y={y + 4} textAnchor="end" className="chart-label">{Math.round(v * 100)}%</text>
            </g>
          );
        })}
        {xTicks.map((month) => {
          const x = PAD_L + (month / 24) * (WIDTH - PAD_L - PAD_R);
          return <text key={month} x={x} y={HEIGHT - 14} textAnchor="middle" className="chart-label">{month}m</text>;
        })}
        <path d={pre} className="chart-line chart-line-pre" />
        {showPost ? <path d={post} className="chart-line chart-line-post" /> : null}
        <path d={selected} className="chart-line chart-line-selected" />
      </svg>
      <div className="chart-legend" aria-hidden="true">
        <span><i className="legend-line pre" />Before newest scan</span>
        <span><i className="legend-line selected" />Selected update</span>
        {showPost ? <span><i className="legend-line post" />Full POST model output</span> : null}
      </div>
    </div>
  );
}
