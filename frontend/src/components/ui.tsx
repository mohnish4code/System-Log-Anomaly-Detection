import type { ReactNode } from "react";
import type { Async } from "../lib";
import type { Outcome } from "../api";

export function Stat({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export function Card({ title, sub, actions, children }: {
  title: string;
  sub?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h2>{title}</h2>
          {sub && <div className="sub">{sub}</div>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Skeleton({ height = 120 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} aria-label="Loading" role="status" />;
}

export function ErrorBox({ error }: { error: Error }) {
  return (
    <div className="error-box" role="alert">
      {error.message}
    </div>
  );
}

/** Render loading / error / ready states of an async value consistently. */
export function Loaded<T>({ state, height, children }: {
  state: Async<T>;
  height?: number;
  children: (data: T) => ReactNode;
}) {
  if (state.status === "loading") return <Skeleton height={height} />;
  if (state.status === "error") return <ErrorBox error={state.error} />;
  return <>{children(state.data)}</>;
}

const AlertIcon = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
    <path d="M8 1.5 15 14H1L8 1.5Z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    <path d="M8 6v3.5M8 11.6v.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const CheckIcon = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
    <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
    <path d="m5 8.2 2 2 4-4.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const MinusIcon = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
    <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
    <path d="M5 8h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

export function OutcomeBadge({ outcome }: { outcome: Outcome }) {
  return outcome === "true_alarm" ? (
    <span className="badge badge-critical"><AlertIcon /> True alarm</span>
  ) : (
    <span className="badge badge-warning"><MinusIcon /> False alarm</span>
  );
}

export function VerdictBadge({ anomalous }: { anomalous: boolean }) {
  return anomalous ? (
    <span className="badge badge-critical"><AlertIcon /> Anomalous</span>
  ) : (
    <span className="badge badge-good"><CheckIcon /> Normal</span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
