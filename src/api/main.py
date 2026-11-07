"""Backend for the web app: scoring plus read endpoints for results, alerts and templates.

    uvicorn src.api.main:app --reload          # API on :8000, docs at /docs
    cd frontend && npm run dev                  # UI on :5173, proxies /api to :8000

After `npm run build`, the API also serves the built UI at http://localhost:8000/.
"""
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints, field_validator

from src.api import data
from src.config import ARTIFACTS_DIR, ROOT
from src.scoring import Scorer

FRONTEND_DIST = ROOT / "frontend" / "dist"

# Limits for POST /api/score.
MAX_LINES = 200_000
MAX_LINE_CHARS = 10_000  # real HDFS/BGL lines are a few hundred characters
MAX_TOTAL_CHARS = 50_000_000
MAX_BODY_BYTES = 64 * 1024 * 1024

app = FastAPI(title="System Log Anomaly Detection",
              description="Score raw HDFS or BGL log lines, explain flagged sessions, and browse results.")
api = APIRouter(prefix="/api")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized bodies from their declared length, before they are read and parsed."""
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": f"Request body exceeds {MAX_BODY_BYTES // (1024 * 1024)} MB"}, 413)
    return await call_next(request)


Dataset = Literal["hdfs", "bgl"]
LogLine = Annotated[str, StringConstraints(max_length=MAX_LINE_CHARS)]


class ScoreRequest(BaseModel):
    dataset: Dataset
    lines: list[LogLine] = Field(min_length=1, max_length=MAX_LINES)
    model: str | None = Field(None, description="defaults to the best model on validation F1")
    window_minutes: int = Field(60, ge=1, description="BGL time-window length")
    top_k: int = Field(3, ge=1, le=20, description="events to report per flagged session")

    @field_validator("lines")
    @classmethod
    def _total_size(cls, lines: list[str]) -> list[str]:
        if sum(map(len, lines)) > MAX_TOTAL_CHARS:
            raise ValueError(f"lines exceed {MAX_TOTAL_CHARS:,} characters in total")
        return lines


class EventContribution(BaseModel):
    event_id: int
    template: str
    count: int
    reason: str


class SessionScore(BaseModel):
    session_id: str
    start_time: int | None
    end_time: int | None
    n_events: int
    score: float
    is_anomaly: bool
    top_events: list[EventContribution]


class ScoreResponse(BaseModel):
    model: str
    threshold: float
    n_lines: int
    n_unparsed: int
    sessions: list[SessionScore]


@lru_cache(maxsize=8)
def get_scorer(dataset: str, model: str | None) -> Scorer:
    if not (ARTIFACTS_DIR / dataset / "classical.joblib").exists():
        raise HTTPException(503, f"No trained artifacts for {dataset}. Run the training pipeline first.")
    try:
        return Scorer(dataset, model)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


_NO_REPLAY = "No replay yet. Run: python -m src.streaming.replay --dataset {dataset}"


def _require_trained(dataset: str) -> None:
    if dataset not in data.trained_datasets():
        raise HTTPException(503, f"No trained artifacts for {dataset}. Run the training pipeline first.")


@api.get("/health")
def health() -> dict:
    return {"status": "ok", "trained": data.trained_datasets()}


@api.get("/datasets")
def datasets() -> list[dict]:
    return [data.overview(d) for d in data.trained_datasets()]


@api.get("/{dataset}/models")
def models(dataset: Dataset) -> dict:
    scorer = get_scorer(dataset, None)
    return {"default": scorer.model_name,
            "models": [{"name": n, "threshold": float(t), "val_f1": float(f)}
                       for n, (_, t, f, _) in scorer.models.items()]}


@api.get("/{dataset}/results")
def results(dataset: Dataset) -> list[dict]:
    _require_trained(dataset)
    return data.results(dataset)


@api.get("/{dataset}/stream")
def stream(dataset: Dataset) -> dict:
    _require_trained(dataset)
    summary = data.stream_summary(dataset)
    if summary is None:
        raise HTTPException(404, _NO_REPLAY.format(dataset=dataset))
    return summary


@api.get("/{dataset}/alerts")
def alerts(dataset: Dataset, outcome: Literal["true_alarm", "false_alarm"] | None = None,
           limit: int = Query(25, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict:
    _require_trained(dataset)
    page = data.alerts(dataset, outcome, limit, offset)
    if page is None:
        raise HTTPException(404, _NO_REPLAY.format(dataset=dataset))
    return page


@api.get("/{dataset}/alerts/{session_id}")
def alert_detail(dataset: Dataset, session_id: str) -> dict:
    _require_trained(dataset)
    if not data.has_stream(dataset):
        raise HTTPException(404, _NO_REPLAY.format(dataset=dataset))
    detail = data.alert_detail(dataset, session_id)
    if detail is None:
        raise HTTPException(404, f"No alert for {session_id}")
    return detail


@api.get("/{dataset}/templates")
def templates(dataset: Dataset) -> dict:
    _require_trained(dataset)
    return data.templates(dataset)


@api.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> dict:
    scorer = get_scorer(req.dataset, req.model)
    return scorer.score_lines(req.lines, window_seconds=req.window_minutes * 60, top_k=req.top_k)


app.include_router(api)

# Serve the built single-page app last, so /api and /docs take precedence.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
