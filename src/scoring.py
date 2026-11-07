"""Load trained artifacts and score raw log lines. Shared by the API, the streaming
replay and the dashboard."""

import pickle
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import ARTIFACTS_DIR
from src.models.sequence import SequenceDetector
from src.parsing.log_formats import BLOCK_RE, LINE_PARSERS, LogRecord

UNMATCHED = -1  # event ID for lines that match no known template


def load_sequence_state(path: Path) -> dict:
    """Load a saved sequence model with weights_only=True (no arbitrary unpickling).

    Older artifacts may hold numpy scalars that the safe loader rejects; those fall back
    to a full unpickle with a warning. Re-train to get a safe artifact.
    """
    try:
        return torch.load(path, weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            f"{path} is not loadable with weights_only=True; falling back to a full unpickle. "
            "Only do this for artifacts you trust, and re-train to upgrade the file.",
            stacklevel=2,
        )
        return torch.load(path, weights_only=False)


@dataclass
class Session:
    session_id: str
    events: list[int] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    start_time: int | None = None
    end_time: int | None = None
    label: int = 0
    first_anomaly_time: int | None = None  # first labelled alert line (BGL only)

    def add(self, event_id: int, rec: LogRecord, raw: str, max_raw: int) -> None:
        self.events.append(event_id)
        if len(self.raw_lines) < max_raw:
            self.raw_lines.append(raw.rstrip("\r\n"))
        self.start_time = rec.timestamp if self.start_time is None else min(self.start_time, rec.timestamp)
        self.end_time = rec.timestamp if self.end_time is None else max(self.end_time, rec.timestamp)
        if rec.label and self.first_anomaly_time is None:
            self.label, self.first_anomaly_time = 1, rec.timestamp


def session_keys(dataset: str, rec: LogRecord, window_seconds: int) -> list[str]:
    if dataset == "hdfs":
        return [f"blk_{b[4:]}" for b in set(BLOCK_RE.findall(rec.content))]
    start = rec.timestamp - rec.timestamp % window_seconds
    return [f"w{start}"]


class Scorer:
    def __init__(self, dataset: str, model: str | None = None):
        self.dataset = dataset
        artifact_dir = ARTIFACTS_DIR / dataset
        self.parser = joblib.load(artifact_dir / "drain.joblib")
        self.templates = {eid: tpl for eid, (tpl, _) in self.parser.templates().items()}

        classical = joblib.load(artifact_dir / "classical.joblib")
        self.vectorizer = classical["vectorizer"]
        self.normal_mean, self.normal_std = classical["normal_mean"], classical["normal_std"]

        # name -> (score function over event sequences, threshold, validation F1, detector)
        self.models: dict[str, tuple] = {}
        for name, entry in classical["models"].items():
            det = entry["model"]
            self.models[name] = (
                lambda seqs, det=det: det.score(self.vectorizer.transform(seqs)),
                entry["threshold"],
                entry["val_f1"],
                det,
            )
        for arch in ("lstm", "transformer"):
            path = artifact_dir / f"{arch}.pt"
            if path.exists():
                state = load_sequence_state(path)
                det = SequenceDetector.from_state(state)
                self.models[det.name] = (det.score, float(state["threshold"]), float(state["val_f1"]), det)

        # Default to the best model on *validation* F1; test results never drive this choice.
        self.model_name = model or max(self.models, key=lambda n: self.models[n][2])
        if self.model_name not in self.models:
            raise ValueError(f"Unknown model {self.model_name!r}; available: {sorted(self.models)}")
        self._score_fn, self.threshold, _, self._detector = self.models[self.model_name]

    def score(self, sequences: Sequence[Sequence[int]]) -> np.ndarray:
        if not sequences:
            return np.empty(0)
        return np.asarray(self._score_fn(list(sequences)), dtype=float)

    def explain(self, events: Sequence[int], top_k: int = 3) -> list[dict]:
        """Events that contributed most to a session's score."""
        if isinstance(self._detector, SequenceDetector):
            _, surprise = self._detector.event_misses(events)
            best: dict[int, float] = {}
            for e, s in zip(events, surprise, strict=True):
                best[e] = max(best.get(e, 0.0), float(s))
            ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
            return [self._describe(e, events, f"unexpected at this point (surprise {s:.1f})") for e, s in ranked]

        counts = np.asarray(self.vectorizer.transform([list(events)]).todense()).ravel()
        z = (counts - self.normal_mean) / (self.normal_std + 0.5)
        z[self.vectorizer.unk_index_] = counts[self.vectorizer.unk_index_] * 10.0  # unseen events dominate
        ids = list(self.vectorizer.event_ids_)
        known = set(ids)
        out = []
        # Events that did happen unusually often (e.g. an exception) are more telling than
        # steps that are missing, so list over-represented events first.
        order = sorted(np.flatnonzero(z), key=lambda c: (z[c] <= 0, -abs(z[c])))
        for col in order[:top_k]:
            if z[col] == 0:
                break
            if col == self.vectorizer.unk_index_:
                unseen = min((e for e in events if e not in known), default=UNMATCHED)
                out.append(
                    {
                        "event_id": int(unseen),
                        "template": self._template(unseen),
                        "count": int(counts[col]),
                        "reason": "event type never seen in training",
                    }
                )
            else:
                direction = "more" if z[col] > 0 else "fewer"
                out.append(
                    self._describe(
                        int(ids[col]), events, f"{direction} than normal (typical {self.normal_mean[col]:.1f})"
                    )
                )
        return out

    def _template(self, event_id: int) -> str:
        return self.templates.get(event_id, "<no matching template>")

    def _describe(self, event_id: int, events: Sequence[int], reason: str) -> dict:
        return {
            "event_id": int(event_id),
            "template": self._template(event_id),
            "count": int(sum(e == event_id for e in events)),
            "reason": reason,
        }

    def group_lines(
        self, lines: Sequence[str], window_seconds: int = 3600, learn: bool = False, max_raw: int = 50
    ) -> tuple[dict[str, Session], int]:
        """Parse and group lines. learn=False matches existing templates only (a request
        must not change the model's parser); learn=True lets Drain add new templates."""
        parse_line = LINE_PARSERS[self.dataset]
        sessions: dict[str, Session] = {}
        unparsed = 0
        for raw in lines:
            rec = parse_line(raw)
            if rec is None:
                unparsed += 1
                continue
            event_id = self.parser.parse(rec.content) if learn else self.parser.match(rec.content)
            event_id = UNMATCHED if event_id is None else event_id
            for key in session_keys(self.dataset, rec, window_seconds):
                sessions.setdefault(key, Session(key)).add(event_id, rec, raw, max_raw)
        return sessions, unparsed

    def score_lines(self, lines: Sequence[str], window_seconds: int = 3600, top_k: int = 3) -> dict:
        sessions, unparsed = self.group_lines(lines, window_seconds)
        ordered = list(sessions.values())
        scores = self.score([s.events for s in ordered])
        results = []
        for s, score in zip(ordered, scores, strict=True):
            flagged = bool(score >= self.threshold)
            results.append(
                {
                    "session_id": s.session_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "n_events": len(s.events),
                    "score": float(score),
                    "is_anomaly": flagged,
                    "top_events": self.explain(s.events, top_k) if flagged else [],
                }
            )
        results.sort(key=lambda r: -r["score"])
        return {
            "model": self.model_name,
            "threshold": float(self.threshold),
            "n_lines": len(lines),
            "n_unparsed": unparsed,
            "sessions": results,
        }
