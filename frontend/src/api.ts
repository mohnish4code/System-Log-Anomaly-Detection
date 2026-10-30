export type DatasetId = "hdfs" | "bgl";

export interface DatasetOverview {
  dataset: DatasetId;
  grouping: string;
  groups: number;
  anomalous: number;
  anomaly_rate: number;
  events: number;
  templates: number;
  start_time: number;
  end_time: number;
}

export interface ModelResult {
  model: string;
  precision: number;
  recall: number;
  f1: number;
  pr_auc: number;
  fp: number;
  tp: number;
  fn: number;
  false_alarms_per_day: number;
  val_f1: number | null;
  threshold: number;
  fit_seconds: number;
  supervised: boolean;
  family: "rule" | "count-vector" | "sequence";
}

export interface TimelinePoint {
  t: number;
  alerts: number;
  anomalous: number;
  groups: number;
  false_alarms: number;
}

export interface StreamSummary {
  groups: number;
  anomalous: number;
  alerts: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  delay_minutes: { median: number; p90: number; max: number } | null;
  bucket_seconds: number;
  timeline: TimelinePoint[];
}

export type Outcome = "true_alarm" | "false_alarm";

export interface AlertRow {
  session_id: string;
  score: number;
  outcome: Outcome;
  n_events: number;
  start_time: number;
  end_time: number;
  alert_time: number;
}

export interface EventContribution {
  event_id: number;
  template: string;
  count: number;
  reason: string;
}

export interface AlertDetail extends AlertRow {
  label: number;
  first_anomaly_time: number | null;
  top_events: EventContribution[];
  raw_lines: string[];
}

export interface TemplateRow {
  event_id: number;
  template: string;
  count: number;
  share: number;
  first_seen: number | null;
}

export interface TemplateInventory {
  total: number;
  train_end: number;
  new_count: number;
  frequent: TemplateRow[];
  new: TemplateRow[];
}

export interface ModelInfo {
  default: string;
  models: { name: string; threshold: number; val_f1: number }[];
}

export interface SessionScore {
  session_id: string;
  start_time: number | null;
  end_time: number | null;
  n_events: number;
  score: number;
  is_anomaly: boolean;
  top_events: EventContribution[];
}

export interface ScoreResponse {
  model: string;
  threshold: number;
  n_lines: number;
  n_unparsed: number;
  sessions: SessionScore[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; trained: DatasetId[] }>("/health"),
  datasets: () => request<DatasetOverview[]>("/datasets"),
  results: (ds: DatasetId) => request<ModelResult[]>(`/${ds}/results`),
  models: (ds: DatasetId) => request<ModelInfo>(`/${ds}/models`),
  stream: (ds: DatasetId) => request<StreamSummary>(`/${ds}/stream`),
  alerts: (ds: DatasetId, outcome: Outcome | null, limit: number, offset: number) => {
    const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (outcome) q.set("outcome", outcome);
    return request<{ total: number; items: AlertRow[] }>(`/${ds}/alerts?${q}`);
  },
  alert: (ds: DatasetId, id: string) => request<AlertDetail>(`/${ds}/alerts/${encodeURIComponent(id)}`),
  templates: (ds: DatasetId) => request<TemplateInventory>(`/${ds}/templates`),
  score: (body: { dataset: DatasetId; lines: string[]; model: string | null }) =>
    request<ScoreResponse>("/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
