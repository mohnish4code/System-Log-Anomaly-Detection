"""Next-event prediction models (DeepLog-style LSTM and a causal Transformer).

The model reads a session's events in order and predicts each next event. An event
is a "miss" when it is not among the model's top-g predictions; sessions with more
misses are more anomalous. Models train on normal sessions only.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

PAD, UNK, BOS = 0, 1, 2
N_SPECIAL = 3


class EventVocab:
    def fit(self, sequences: Sequence[Sequence[int]]) -> "EventVocab":
        events = sorted({int(e) for s in sequences for e in s})
        self.index_ = {e: i + N_SPECIAL for i, e in enumerate(events)}
        return self

    def __len__(self) -> int:
        return len(self.index_) + N_SPECIAL

    def encode(self, sequence: Sequence[int]) -> np.ndarray:
        return np.fromiter((self.index_.get(int(e), UNK) for e in sequence), dtype=np.int64, count=len(sequence))


class NextEventLSTM(nn.Module):
    def __init__(
        self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2, max_len: int = 128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(self.embed(x))
        return self.out(hidden)


class NextEventTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        return self.out(self.encoder(h, mask=causal, is_causal=True, src_key_padding_mask=x == PAD))


ARCHITECTURES = {"lstm": NextEventLSTM, "transformer": NextEventTransformer}


def make_chunks(encoded: np.ndarray, max_len: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split one encoded session into (input, target) pairs of at most max_len steps.
    The first input starts with BOS; later chunks start with the previous event."""
    shifted = np.r_[BOS, encoded[:-1]]
    return [(shifted[i : i + max_len], encoded[i : i + max_len]) for i in range(0, len(encoded), max_len)]


def _pad_batch(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(x) for x, _ in pairs)
    xs = np.zeros((len(pairs), T), dtype=np.int64)
    ys = np.zeros((len(pairs), T), dtype=np.int64)
    for i, (x, y) in enumerate(pairs):
        xs[i, : len(x)] = x
        ys[i, : len(y)] = y
    return torch.from_numpy(xs), torch.from_numpy(ys)


def _length_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    # Grouping similar lengths keeps padding (wasted compute) low.
    order = np.argsort(lengths, kind="stable")
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    return batches


class SequenceDetector:
    supervised = False

    def __init__(
        self,
        arch: str = "lstm",
        top_g: int = 9,
        max_len: int = 128,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.arch, self.top_g, self.max_len = arch, top_g, max_len
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.name = {"lstm": "LSTM (DeepLog-style)", "transformer": "Transformer"}[arch]

    def fit(self, sequences: Sequence[Sequence[int]], log=print) -> "SequenceDetector":
        """sequences: normal training sessions. Duplicates are dropped: HDFS has a few thousand
        distinct normal sessions repeated hundreds of thousands of times."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        unique = list({tuple(s) for s in sequences if len(s)})
        self.vocab_ = EventVocab().fit(unique)
        self.model_ = ARCHITECTURES[self.arch](len(self.vocab_), max_len=self.max_len)

        pairs = [p for s in unique for p in make_chunks(self.vocab_.encode(s), self.max_len)]
        lengths = np.array([len(x) for x, _ in pairs])
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        log(f"{len(unique):,} distinct normal sessions -> {len(pairs):,} chunks, vocab {len(self.vocab_)}")

        self.model_.train()
        for epoch in range(self.epochs):
            total, n = 0.0, 0
            for idx in _length_batches(lengths, self.batch_size, rng):
                x, y = _pad_batch([pairs[i] for i in idx])
                logits = self.model_(x)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
                n += len(idx)
            log(f"epoch {epoch + 1}/{self.epochs}  loss {total / n:.4f}")
        self.model_.eval()
        return self

    @torch.no_grad()
    def event_misses(self, sequence: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Per-event (is_miss, surprise = -log p) for one session, used to explain alerts."""
        encoded = self.vocab_.encode(sequence)
        misses, surprise = [], []
        for x, y in make_chunks(encoded, self.max_len):
            m, s = self._score_chunks([(x, y)])
            misses.append(m[0][: len(y)])
            surprise.append(s[0][: len(y)])
        return np.concatenate(misses), np.concatenate(surprise)

    def _score_chunks(self, pairs) -> tuple[np.ndarray, np.ndarray]:
        x, y = _pad_batch(pairs)
        log_probs = torch.log_softmax(self.model_(x), dim=-1)
        target_lp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        # Rank of the true event = number of events the model rated strictly more likely.
        rank = (log_probs > target_lp.unsqueeze(-1)).sum(-1)
        miss = (rank >= self.top_g) | (y == UNK)
        surprise = torch.where(y == UNK, torch.full_like(target_lp, 30.0), -target_lp)
        valid = y != PAD
        return (miss & valid).numpy(), torch.where(valid, surprise, 0.0).numpy()

    @torch.no_grad()
    def score(self, sequences: Sequence[Sequence[int]]) -> np.ndarray:
        """Score = number of missed events, with ties broken by the single most surprising
        event (scaled into [0, 1) so it never outweighs a whole miss)."""
        keys = [tuple(s) for s in sequences]
        unique = list(dict.fromkeys(keys))
        owner, pairs = [], []
        for u, seq in enumerate(unique):
            for p in make_chunks(self.vocab_.encode(seq), self.max_len) if seq else []:
                owner.append(u)
                pairs.append(p)
        owner = np.array(owner, dtype=np.int64)
        lengths = np.array([len(x) for x, _ in pairs])

        n_miss = np.zeros(len(unique))
        max_surprise = np.zeros(len(unique))
        for idx in _length_batches(lengths, 512):
            miss, surprise = self._score_chunks([pairs[i] for i in idx])
            np.add.at(n_miss, owner[idx], miss.sum(1))
            np.maximum.at(max_surprise, owner[idx], surprise.max(1))

        per_unique = n_miss + max_surprise / (1.0 + max_surprise)
        lookup = dict(zip(unique, per_unique, strict=True))
        return np.array([lookup[k] for k in keys])

    def state(self) -> dict:
        return {
            "arch": self.arch,
            "top_g": self.top_g,
            "max_len": self.max_len,
            "vocab": self.vocab_.index_,
            "weights": self.model_.state_dict(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "SequenceDetector":
        det = cls(arch=state["arch"], top_g=state["top_g"], max_len=state["max_len"])
        det.vocab_ = EventVocab()
        det.vocab_.index_ = state["vocab"]
        det.model_ = ARCHITECTURES[det.arch](len(det.vocab_), max_len=det.max_len)
        det.model_.load_state_dict(state["weights"])
        det.model_.eval()
        return det
