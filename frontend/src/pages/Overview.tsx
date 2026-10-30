import { useMemo, useState } from "react";
import { api, type DatasetOverview, type ModelResult } from "../api";
import { METRIC_LABEL, ModelChart, type Metric } from "../components/charts";
import { Card, Loaded, Stat } from "../components/ui";
import { fmt, useAsync, type Theme } from "../lib";

type SortKey = "model" | "precision" | "recall" | "f1" | "pr_auc" | "fp" | "val_f1";

export function Overview({ overview, theme }: { overview: DatasetOverview; theme: Theme }) {
  const ds = overview.dataset;
  const results = useAsync(() => api.results(ds), [ds]);
  const [metric, setMetric] = useState<Metric>("f1");

  return (
    <>
      <div className="page-head">
        <h1>{ds.toUpperCase()} overview</h1>
        <p>
          {fmt.compact(overview.events)} log lines parsed into {overview.templates} templates and grouped into{" "}
          {fmt.int(overview.groups)} {overview.grouping}, {fmt.date(overview.start_time)} to{" "}
          {fmt.date(overview.end_time)}.
        </p>
      </div>

      <div className="stats">
        <Stat label="Log lines" value={fmt.compact(overview.events)} />
        <Stat label={ds === "hdfs" ? "Sessions" : "Time windows"} value={fmt.int(overview.groups)} />
        <Stat label="Anomalous" value={fmt.pct(overview.anomaly_rate)} note={`${fmt.int(overview.anomalous)} groups`} />
        <Stat label="Log templates" value={fmt.int(overview.templates)} note="found by Drain" />
      </div>

      <Card
        title="Model comparison"
        sub="Test split · thresholds tuned on validation, then frozen"
        actions={
          <div className="segmented" role="group" aria-label="Metric">
            {(Object.keys(METRIC_LABEL) as Metric[]).map((m) => (
              <button key={m} aria-pressed={metric === m} onClick={() => setMetric(m)}>
                {METRIC_LABEL[m]}
              </button>
            ))}
          </div>
        }
      >
        <Loaded state={results} height={300}>
          {(rows) => <ModelChart results={rows} metric={metric} theme={theme} />}
        </Loaded>
      </Card>

      <Card title="All metrics" sub="Click a column to sort. Supervised models use labels, so they show the ceiling.">
        <Loaded state={results} height={240}>{(rows) => <ResultsTable rows={rows} />}</Loaded>
      </Card>
    </>
  );
}

const COLUMNS: { key: SortKey; label: string; digits?: number }[] = [
  { key: "model", label: "Model" },
  { key: "precision", label: "Precision", digits: 3 },
  { key: "recall", label: "Recall", digits: 3 },
  { key: "f1", label: "F1", digits: 3 },
  { key: "pr_auc", label: "PR-AUC", digits: 3 },
  { key: "val_f1", label: "Val F1", digits: 3 },
  { key: "fp", label: "False alarms" },
];

function ResultsTable({ rows }: { rows: ModelResult[] }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "f1", desc: true });
  const sorted = useMemo(() => {
    const out = [...rows].sort((a, b) => {
      const x = a[sort.key] ?? -1;
      const y = b[sort.key] ?? -1;
      return x < y ? -1 : x > y ? 1 : 0;
    });
    return sort.desc ? out.reverse() : out;
  }, [rows, sort]);
  const best = Math.max(...rows.map((r) => r.f1));

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {COLUMNS.map((c) => (
              <th key={c.key} className={c.key === "model" ? "" : "num"}
                aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}>
                <button onClick={() => setSort({ key: c.key, desc: sort.key === c.key ? !sort.desc : c.key !== "model" })}>
                  {c.label} {sort.key === c.key ? (sort.desc ? "↓" : "↑") : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.model}>
              <td>
                {r.model.replace(" (ERROR/FATAL/fail/exception)", "").replace(" (supervised)", "")}{" "}
                {r.supervised && <span className="badge badge-neutral">supervised</span>}
                {r.family === "sequence" && <span className="badge badge-neutral">sequence</span>}
                {r.family === "rule" && <span className="badge badge-neutral">rule</span>}
              </td>
              <td className="num">{fmt.ratio(r.precision)}</td>
              <td className="num">{fmt.ratio(r.recall)}</td>
              <td className={`num ${r.f1 === best ? "best" : ""}`}>{fmt.ratio(r.f1)}</td>
              <td className="num">{fmt.ratio(r.pr_auc)}</td>
              <td className="num">{fmt.ratio(r.val_f1)}</td>
              <td className="num">{fmt.int(r.fp)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
