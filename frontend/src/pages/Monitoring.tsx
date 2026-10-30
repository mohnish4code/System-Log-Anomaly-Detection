import { useEffect, useState } from "react";
import { api, ApiError, type AlertRow, type DatasetId, type Outcome } from "../api";
import { Timeline } from "../components/charts";
import { Card, Empty, ErrorBox, Loaded, OutcomeBadge, Skeleton, Stat } from "../components/ui";
import { fmt, useAsync, type Theme } from "../lib";

const PAGE = 15;

export function Monitoring({ ds, theme }: { ds: DatasetId; theme: Theme }) {
  const stream = useAsync(() => api.stream(ds), [ds]);

  if (stream.status === "error" && stream.error instanceof ApiError && stream.error.status === 404) {
    return (
      <>
        <Head ds={ds} />
        <Empty>
          No replay has been run for {ds.toUpperCase()} yet. Run{" "}
          <code>python -m src.streaming.replay --dataset {ds}</code>
        </Empty>
      </>
    );
  }

  return (
    <>
      <Head ds={ds} />
      <Loaded state={stream} height={110}>
        {(s) => (
          <div className="stats">
            <Stat label="Groups scored" value={fmt.int(s.groups)} note={`${fmt.int(s.anomalous)} truly anomalous`} />
            <Stat label="Alerts raised" value={fmt.int(s.alerts)} note={`${fmt.int(s.fp)} false alarms`} />
            <Stat label="Precision" value={fmt.ratio(s.precision)} note="alerts that were real" />
            <Stat label="Recall" value={fmt.ratio(s.recall)} note={`${fmt.int(s.fn)} anomalies missed`} />
            <Stat label="Median detection delay"
              value={s.delay_minutes ? fmt.minutes(s.delay_minutes.median) : "–"}
              note={s.delay_minutes ? `p90 ${fmt.minutes(s.delay_minutes.p90)}` : undefined} />
          </div>
        )}
      </Loaded>

      <Card title="Alerts over time" sub={ds === "hdfs" ? "Per hour of session start (log time)" : "Per day (log time)"}>
        <Loaded state={stream} height={300}>
          {(s) => <Timeline points={s.timeline} bucketSeconds={s.bucket_seconds} theme={theme} />}
        </Loaded>
      </Card>

      {stream.status === "ready" && <AlertExplorer ds={ds} />}
    </>
  );
}

function Head({ ds }: { ds: DatasetId }) {
  return (
    <div className="page-head">
      <h1>Live monitoring replay</h1>
      <p>
        The final 20% of the {ds.toUpperCase()} log replayed line by line, as a running system would emit it.
        Lines are parsed online, grouped, and scored; each alert links back to the raw lines behind it.
      </p>
    </div>
  );
}

function AlertExplorer({ ds }: { ds: DatasetId }) {
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const page = useAsync(() => api.alerts(ds, outcome, PAGE, offset), [ds, outcome, offset]);

  useEffect(() => {
    setOffset(0);
    setSelected(null);
  }, [ds, outcome]);

  useEffect(() => {
    if (page.status === "ready" && !selected && page.data.items.length) setSelected(page.data.items[0].session_id);
  }, [page, selected]);

  return (
    <div className="grid-split" style={{ marginTop: 16 }}>
      <Card
        title="Alerts"
        sub="Highest score first"
        actions={
          <div className="segmented" role="group" aria-label="Filter alerts">
            {([[null, "All"], ["true_alarm", "True"], ["false_alarm", "False"]] as const).map(([v, label]) => (
              <button key={label} aria-pressed={outcome === v} onClick={() => setOutcome(v)}>{label}</button>
            ))}
          </div>
        }
      >
        <Loaded state={page} height={420}>
          {({ total, items }) =>
            items.length === 0 ? (
              <Empty>No alerts match this filter.</Empty>
            ) : (
              <>
                <AlertTable items={items} selected={selected} onSelect={setSelected} />
                <div className="pager">
                  <span>{fmt.int(offset + 1)}–{fmt.int(offset + items.length)} of {fmt.int(total)}</span>
                  <div className="toolbar">
                    <button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
                      Previous
                    </button>
                    <button className="btn" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>
                      Next
                    </button>
                  </div>
                </div>
              </>
            )
          }
        </Loaded>
      </Card>
      {selected ? <AlertDetailCard ds={ds} id={selected} /> : <Card title="Alert detail"><Empty>Select an alert.</Empty></Card>}
    </div>
  );
}

function AlertTable({ items, selected, onSelect }: {
  items: AlertRow[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Group</th>
            <th>Outcome</th>
            <th className="num">Score</th>
            <th className="num">Events</th>
          </tr>
        </thead>
        <tbody>
          {items.map((a) => (
            <tr key={a.session_id} className="clickable" aria-selected={a.session_id === selected}
              tabIndex={0} onClick={() => onSelect(a.session_id)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(a.session_id)}>
              <td>
                <div className="mono">{a.session_id}</div>
                <div className="muted" style={{ fontSize: 12 }}>alerted {fmt.dateTime(a.alert_time)}</div>
              </td>
              <td><OutcomeBadge outcome={a.outcome} /></td>
              <td className="num">{a.score.toFixed(3)}</td>
              <td className="num">{fmt.int(a.n_events)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AlertDetailCard({ ds, id }: { ds: DatasetId; id: string }) {
  const detail = useAsync(() => api.alert(ds, id), [ds, id]);
  if (detail.status === "loading") return <section className="card"><Skeleton height={420} /></section>;
  if (detail.status === "error") return <section className="card"><ErrorBox error={detail.error} /></section>;
  const a = detail.data;
  const delay = (a.alert_time - (a.first_anomaly_time ?? a.start_time)) / 60;

  return (
    <Card title={a.session_id} sub={`${fmt.int(a.n_events)} events · started ${fmt.dateTime(a.start_time)}`}
      actions={<OutcomeBadge outcome={a.outcome} />}>
      <div className="stats" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
        <Stat label="Score" value={a.score.toFixed(3)} />
        <Stat label="Alerted" value={fmt.hour(a.alert_time)} />
        <Stat label="After start" value={fmt.minutes(Math.max(delay, 0))} />
      </div>
      <h3 style={{ fontSize: 13.5, margin: "4px 0 2px" }}>Why it was flagged</h3>
      <ul className="event-list">
        {a.top_events.map((e) => (
          <li key={e.event_id}>
            <div className="template">{e.template}</div>
            <div className="reason">{e.reason} · seen {e.count}×</div>
          </li>
        ))}
      </ul>
      <h3 style={{ fontSize: 13.5, margin: "14px 0 8px" }}>
        Raw log lines <span className="muted" style={{ fontWeight: 400 }}>(first {a.raw_lines.length})</span>
      </h3>
      <div className="code-block">{a.raw_lines.join("\n")}</div>
    </Card>
  );
}
