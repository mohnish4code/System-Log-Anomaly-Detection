import { api, type DatasetId, type TemplateRow } from "../api";
import { Card, Empty, Loaded, Stat } from "../components/ui";
import { fmt, useAsync } from "../lib";

export function Templates({ ds }: { ds: DatasetId }) {
  const inv = useAsync(() => api.templates(ds), [ds]);

  return (
    <>
      <div className="page-head">
        <h1>Log templates</h1>
        <p>
          Drain turns each raw line into a fixed template plus variable values (block IDs, IPs, numbers).
          Templates that first appear after the training period are unknown to every model.
        </p>
      </div>
      <Loaded state={inv} height={400}>
        {(t) => {
          const top = t.frequent[0]?.share ?? 1;
          return (
            <>
              <div className="stats">
                <Stat label="Templates" value={fmt.int(t.total)} />
                <Stat label="First seen after training" value={fmt.int(t.new_count)}
                  note={`training ended ${fmt.date(t.train_end)}`} />
                <Stat label="Most common template" value={fmt.pct(top)} note="of all lines" />
              </div>
              <div className="grid-2">
                <Card title="Most frequent" sub={`Top ${t.frequent.length}`}>
                  <TemplateTable rows={t.frequent} showShare maxShare={top} />
                </Card>
                <Card title="Newly seen" sub="Appeared only after the training period">
                  {t.new.length ? <TemplateTable rows={t.new} showFirstSeen /> : <Empty>None.</Empty>}
                </Card>
              </div>
            </>
          );
        }}
      </Loaded>
    </>
  );
}

function TemplateTable({ rows, showShare, showFirstSeen, maxShare = 1 }: {
  rows: TemplateRow[];
  showShare?: boolean;
  showFirstSeen?: boolean;
  maxShare?: number;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Template</th>
            <th className="num">Lines</th>
            {showShare && <th>Share</th>}
            {showFirstSeen && <th>First seen</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.event_id}>
              <td className="mono muted">E{r.event_id}</td>
              <td className="template">{r.template}</td>
              <td className="num">{fmt.int(r.count)}</td>
              {showShare && (
                <td style={{ verticalAlign: "middle" }} title={fmt.pct(r.share, 2)}>
                  <div className="share-bar"><span style={{ width: `${(r.share / maxShare) * 100}%` }} /></div>
                </td>
              )}
              {showFirstSeen && <td className="muted" style={{ whiteSpace: "nowrap" }}>{r.first_seen ? fmt.dateTime(r.first_seen) : "–"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
