import { useEffect, useState } from "react";
import { api, type DatasetId } from "./api";
import { Empty, ErrorBox, Skeleton } from "./components/ui";
import { useAsync, useTheme } from "./lib";
import { Analyze } from "./pages/Analyze";
import { Monitoring } from "./pages/Monitoring";
import { Overview } from "./pages/Overview";
import { Templates } from "./pages/Templates";

const PAGES = [
  { id: "overview", label: "Overview" },
  { id: "monitoring", label: "Monitoring" },
  { id: "templates", label: "Templates" },
  { id: "analyze", label: "Analyze logs" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

/** Route lives in the hash (#/bgl/monitoring) so views are linkable and survive reloads. */
function readHash(): { ds: DatasetId | null; page: PageId } {
  const [ds, page] = window.location.hash.replace(/^#\/?/, "").split("/");
  return {
    ds: ds === "hdfs" || ds === "bgl" ? ds : null,
    page: PAGES.some((p) => p.id === page) ? (page as PageId) : "overview",
  };
}

export function App() {
  const [theme, setTheme] = useTheme();
  const datasets = useAsync(() => api.datasets(), []);
  const [route, setRoute] = useState(readHash);

  useEffect(() => {
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const available = datasets.status === "ready" ? datasets.data.map((d) => d.dataset) : [];
  const ds: DatasetId | null = route.ds && available.includes(route.ds) ? route.ds : available[0] ?? null;
  const go = (next: { ds?: DatasetId; page?: PageId }) => {
    window.location.hash = `/${next.ds ?? ds}/${next.page ?? route.page}`;
  };
  const overview = datasets.status === "ready" ? datasets.data.find((d) => d.dataset === ds) : undefined;

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 32 32">
                <path d="M5 20h5l3.5-10 4.5 14 3.5-8H27" fill="none" stroke="currentColor" strokeWidth="2.8"
                  strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            Log Anomaly Detection
          </div>
          {available.length > 0 && (
            <div className="segmented" role="group" aria-label="Dataset">
              {available.map((d) => (
                <button key={d} aria-pressed={d === ds} onClick={() => go({ ds: d })}>{d.toUpperCase()}</button>
              ))}
            </div>
          )}
          <button className="icon-btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title="Toggle theme">
            {theme === "dark" ? (
              <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" /></svg>
            )}
          </button>
        </div>
        <nav className="nav" aria-label="Sections">
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${ds ?? "hdfs"}/${p.id}`} aria-current={p.id === route.page ? "page" : undefined}>
              {p.label}
            </a>
          ))}
        </nav>
      </header>

      <main>
        {datasets.status === "loading" && <Skeleton height={400} />}
        {datasets.status === "error" && (
          <ErrorBox error={new Error(`Cannot reach the API (${datasets.error.message}). Is uvicorn running on port 8000?`)} />
        )}
        {datasets.status === "ready" && !ds && (
          <Empty>No trained models found. Run the training pipeline described in the README.</Empty>
        )}
        {ds && overview && (
          <>
            {route.page === "overview" && <Overview overview={overview} theme={theme} />}
            {route.page === "monitoring" && <Monitoring ds={ds} theme={theme} />}
            {route.page === "templates" && <Templates ds={ds} />}
            {route.page === "analyze" && <Analyze ds={ds} />}
          </>
        )}
      </main>
    </>
  );
}
