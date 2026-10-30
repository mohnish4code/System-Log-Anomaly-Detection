"""Event count vectors: how often each event type occurs in a session."""
from collections.abc import Sequence

import numpy as np
from scipy import sparse


class EventCountVectorizer:
    """Vocabulary is learned from training sessions only. Event IDs first seen later
    (new templates in the test period) are counted in a single trailing UNK column."""

    def fit(self, sequences: Sequence[Sequence[int]]) -> "EventCountVectorizer":
        seen = np.unique(np.concatenate([np.asarray(s, dtype=np.int64) for s in sequences]))
        self.event_ids_ = seen
        self.unk_index_ = len(seen)
        self._lookup = np.full(int(seen.max()) + 1, self.unk_index_, dtype=np.int64)
        self._lookup[seen] = np.arange(len(seen))
        return self

    @property
    def n_features(self) -> int:
        return self.unk_index_ + 1

    def feature_names(self, templates: dict[int, str] | None = None) -> list[str]:
        templates = templates or {}
        return [templates.get(int(e), f"E{e}") for e in self.event_ids_] + ["<UNSEEN>"]

    def transform(self, sequences: Sequence[Sequence[int]]) -> sparse.csr_matrix:
        lengths = np.fromiter(map(len, sequences), dtype=np.int64, count=len(sequences))
        flat = np.concatenate([np.asarray(s, dtype=np.int64) for s in sequences]) if lengths.sum() else \
            np.empty(0, dtype=np.int64)
        in_range = flat < len(self._lookup)
        cols = np.full(len(flat), self.unk_index_, dtype=np.int64)
        cols[in_range] = self._lookup[flat[in_range]]
        rows = np.repeat(np.arange(len(sequences)), lengths)
        return sparse.csr_matrix((np.ones(len(flat), dtype=np.float64), (rows, cols)),
                                 shape=(len(sequences), self.n_features))

    def fit_transform(self, sequences: Sequence[Sequence[int]]) -> sparse.csr_matrix:
        return self.fit(sequences).transform(sequences)
