import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ModelResult, TimelinePoint } from "../api";
import { CHART, fmt, type Theme } from "../lib";

export type Metric = "f1" | "pr_auc" | "precision" | "recall";
export const METRIC_LABEL: Record<Metric, string> = {
  f1: "F1",
  pr_auc: "PR-AUC",
  precision: "Precision",
  recall: "Recall",
};

const shortName = (m: string) =>
  m.replace(" (ERROR/FATAL/fail/exception)", "").replace(" (supervised)", " (sup.)");

export function ModelChart({ results, metric, theme }: { results: ModelResult[]; metric: Metric; theme: Theme }) {
  const c = CHART[theme];
  const data = [...results]
    .sort((a, b) => b[metric] - a[metric])
    .map((r) => ({ ...r, name: shortName(r.model), value: r[metric] }));

  return (
    <ResponsiveContainer width="100%" height={data.length * 36 + 40}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }}>
        <CartesianGrid horizontal={false} stroke={c.grid} />
        <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fill: c.axis, fontSize: 12 }}
          axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" width={170} tick={{ fill: c.axis, fontSize: 12.5 }}
          axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: c.cursor, opacity: 0.25 }} content={<ModelTooltip />} />
        <Bar dataKey="value" fill={c.series1} radius={[0, 4, 4, 0]} barSize={18} isAnimationActive={false}>
          <LabelList dataKey="value" position="right" formatter={(v: number) => v.toFixed(3)}
            style={{ fill: "var(--ink-2)", fontSize: 12 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

interface TipProps<T> {
  active?: boolean;
  payload?: { payload: T; color?: string; name?: string; value?: number }[];
}

function ModelTooltip({ active, payload }: TipProps<ModelResult>) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{r.model}</div>
      {(["precision", "recall", "f1", "pr_auc"] as Metric[]).map((m) => (
        <div className="row" key={m}>
          {METRIC_LABEL[m]} <b>{r[m].toFixed(3)}</b>
        </div>
      ))}
      <div className="row">False alarms <b>{fmt.int(r.fp)}</b></div>
    </div>
  );
}

export function Timeline({ points, bucketSeconds, theme }: {
  points: TimelinePoint[];
  bucketSeconds: number;
  theme: Theme;
}) {
  const c = CHART[theme];
  const tick = bucketSeconds >= 86400 ? fmt.date : fmt.hour;
  return (
    <>
      <div className="legend" style={{ marginBottom: 8 }}>
        <span><i style={{ background: c.series1 }} /> Alerts raised</span>
        <span><i style={{ background: c.series2 }} /> Truly anomalous</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={c.grid} />
          <XAxis dataKey="t" tickFormatter={tick} tick={{ fill: c.axis, fontSize: 12 }} minTickGap={40}
            axisLine={{ stroke: c.grid }} tickLine={false} />
          <YAxis tick={{ fill: c.axis, fontSize: 12 }} axisLine={false} tickLine={false} width={48}
            tickFormatter={fmt.compact} />
          <Tooltip cursor={{ stroke: c.cursor }} content={<TimelineTooltip format={tick} />} />
          <Line type="monotone" dataKey="alerts" name="Alerts raised" stroke={c.series1} strokeWidth={2}
            dot={false} activeDot={{ r: 5, strokeWidth: 2, stroke: "var(--surface)" }} isAnimationActive={false} />
          <Line type="monotone" dataKey="anomalous" name="Truly anomalous" stroke={c.series2} strokeWidth={2}
            dot={false} activeDot={{ r: 5, strokeWidth: 2, stroke: "var(--surface)" }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

function TimelineTooltip({ active, payload, format }: TipProps<TimelinePoint> & { format: (s: number) => string }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{format(p.t)} UTC</div>
      {payload.map((s) => (
        <div className="row" key={s.name}>
          <span className="dot" style={{ background: s.color }} /> {s.name} <b>{fmt.int(s.value ?? 0)}</b>
        </div>
      ))}
      <div className="row muted">Groups scored <b>{fmt.int(p.groups)}</b></div>
      <div className="row muted">False alarms <b>{fmt.int(p.false_alarms)}</b></div>
    </div>
  );
}
