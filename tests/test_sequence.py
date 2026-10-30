import numpy as np
import pytest

from src.models.sequence import BOS, UNK, SequenceDetector, make_chunks


def test_make_chunks_shifts_inputs_and_keeps_context():
    chunks = make_chunks(np.array([5, 6, 7, 8, 9]), max_len=2)
    assert [(x.tolist(), y.tolist()) for x, y in chunks] == [
        ([BOS, 5], [5, 6]), ([6, 7], [7, 8]), ([8], [9])]


@pytest.mark.parametrize("arch", ["lstm", "transformer"])
def test_detector_flags_out_of_order_and_unseen_events(arch):
    # Normal behaviour: a fixed workflow 1 -> 2 -> 3 -> 4, repeated.
    normal = [[1, 2, 3, 4] * k for k in range(1, 6)]
    det = SequenceDetector(arch=arch, top_g=1, max_len=16, epochs=150, batch_size=8, lr=1e-2)
    det.fit(normal, log=lambda *_: None)

    scores = det.score([[1, 2, 3, 4], [1, 3, 2, 4], [1, 2, 99, 4], []])
    assert scores[0] < 1 <= scores[1]  # skipped / reordered steps are misses
    assert scores[2] >= 1  # unseen event (UNK) is always a miss
    assert scores[3] == 0

    misses, surprise = det.event_misses([1, 2, 99, 4])
    assert misses.tolist()[2] and det.vocab_.encode([99])[0] == UNK
    assert surprise.argmax() == 2


def test_state_round_trip():
    det = SequenceDetector(arch="lstm", max_len=8, epochs=1).fit([[1, 2, 3]], log=lambda *_: None)
    clone = SequenceDetector.from_state(det.state())
    seqs = [[1, 2, 3], [3, 2, 1]]
    assert np.allclose(det.score(seqs), clone.score(seqs))
