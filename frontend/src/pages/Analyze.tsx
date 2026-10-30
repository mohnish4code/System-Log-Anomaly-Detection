import { useEffect, useState } from "react";
import { api, type DatasetId, type ScoreResponse } from "../api";
import { Card, Empty, ErrorBox, VerdictBadge } from "../components/ui";
import examples from "../examples.json";
import { fmt, useAsync } from "../lib";

const PLACEHOLDER: Record<DatasetId, string> = {
  hdfs: "081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_38865049064139660 of size 67108864 from /10.251.42.84",
  bgl: "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected",
};

export function Analyze({ ds }: { ds: DatasetId }) {
  const models = useAsync(() => api.models(ds), [ds]);
  const [text, setText] = useState("");
  const [model, setModel] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setText("");
    setModel("");
    setResult(null);
    setError(null);
  }, [ds]);

  const lines = text.split(/\r?\n/).filter((l) => l.trim());

  async function run() {
    setRunning(true);
    setError(null);
    try {
      setResult(await api.score({ dataset: ds, lines, model: model || null }));
    } catch (e) {
      setError(e as Error);
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Analyze logs</h1>
        <p>
          Paste raw {ds.toUpperCase()} log lines. They are matched against known templates, grouped into{" "}
          {ds === "hdfs" ? "block sessions" : "hourly windows"}, scored, and each flag is explained.
        </p>
      </div>

      <Card
        title="Log lines"
        sub={`${fmt.int(lines.length)} line${lines.length === 1 ? "" : "s"}`}
        actions={
          <div className="toolbar">
            <span className="muted" style={{ fontSize: 12.5 }}>Load example:</span>
            <button className="btn" onClick={() => setText(examples[ds].normal.join("\n"))}>Normal</button>
            <button className="btn" onClick={() => setText(examples[ds].anomalous.join("\n"))}>Anomalous</button>
          </div>
        }
      >
        <label htmlFor="log-input" className="muted" style={{ fontSize: 12.5 }}>One log line per row</label>
        <textarea id="log-input" rows={10} value={text} placeholder={PLACEHOLDER[ds]} spellCheck={false}
          onChange={(e) => setText(e.target.value)} />
        <div className="toolbar" style={{ marginTop: 12, justifyContent: "space-between" }}>
          <label className="toolbar" style={{ gap: 8 }}>
            <span className="muted" style={{ fontSize: 12.5 }}>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="">
                {models.status === "ready" ? `Default · ${models.data.default}` : "Default"}
              </option>
              {models.status === "ready" &&
                models.data.models.map((m) => (
                  <option key={m.name} value={m.name}>{m.name} (val F1 {m.val_f1.toFixed(3)})</option>
                ))}
            </select>
          </label>
          <div className="toolbar">
            <button className="btn" onClick={() => { setText(""); setResult(null); }} disabled={!text}>Clear</button>
            <button className="btn btn-primary" onClick={run} disabled={!lines.length || running}>
              {running ? "Scoring…" : "Score lines"}
            </button>
          </div>
        </div>
      </Card>

      {error && <div style={{ marginTop: 16 }}><ErrorBox error={error} /></div>}
      {result && <Results result={result} />}
    </>
  );
}

function Results({ result }: { result: ScoreResponse }) {
  const flagged = result.sessions.filter((s) => s.is_anomaly).length;
  return (
    <Card
      title={`${fmt.int(result.sessions.length)} group${result.sessions.length === 1 ? "" : "s"} scored · ${flagged} flagged`}
      sub={`${result.model} · threshold ${result.threshold.toPrecision(4)} · ${result.n_unparsed} of ${result.n_lines} lines unparseable`}
    >
      {result.sessions.length === 0 ? (
        <Empty>No line matched the {`log format`}. Check the dataset selected at the top.</Empty>
      ) : (
        result.sessions.map((s) => (
          <div className="result" key={s.session_id}>
            <div className="result-head">
              <VerdictBadge anomalous={s.is_anomaly} />
              <span className="mono">{s.session_id}</span>
              <span className="muted" style={{ marginLeft: "auto", fontSize: 12.5 }}>
                score <b style={{ color: "var(--ink)" }}>{s.score.toFixed(3)}</b> · {fmt.int(s.n_events)} events
              </span>
            </div>
            {s.top_events.length > 0 && (
              <ul className="event-list" style={{ marginTop: 6 }}>
                {s.top_events.map((e) => (
                  <li key={e.event_id}>
                    <div className="template">{e.template}</div>
                    <div className="reason">{e.reason} · seen {e.count}×</div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))
      )}
      <div className="footer-note">
        Short pastes are scored as complete groups, so a session cut off mid-way can look abnormal.
      </div>
    </Card>
  );
}
