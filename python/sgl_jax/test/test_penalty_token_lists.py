"""Ragged token feeding (speculative rounds append several tokens per request) must count
exactly like feeding the same tokens one step at a time."""

from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sgl_jax.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sgl_jax.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer

V = 16


class _Info:  # weak-referenceable stand-in for ScheduleReqsInfo
    def __init__(self, reqs):
        self.reqs = reqs


def _orch():
    reqs = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(frequency_penalty=0.5, presence_penalty=0.25)
        ),
        SimpleNamespace(
            sampling_params=SimpleNamespace(frequency_penalty=1.0, presence_penalty=0.0)
        ),
    ]
    info = _Info(reqs)
    orch = BatchedPenalizerOrchestrator(
        V, info, {BatchedFrequencyPenalizer, BatchedPresencePenalizer}
    )
    return orch, info  # keep info alive (the orchestrator holds a weakref)


def test_ragged_matches_stepwise():
    lists = [[5, 5, 7], [9]]
    a, info_a = _orch()
    a.cumulate_output_token_lists(lists)
    b, info_b = _orch()
    for step in range(3):  # row 1 has one token; pad its later steps with a token, then undo
        b.cumulate_output_tokens(np.array([lists[0][step], lists[1][0] if step == 0 else 0]))
    pen_b = b.apply()
    pen_b[1, 0] = 0.0  # padding token 0 fed to row 1 at steps 1-2 (frequency 2 x -1.0)
    np.testing.assert_allclose(a.apply(), pen_b)
    pen = a.apply()
    assert pen[0, 5] == -(2 * 0.5 + 0.25) and pen[0, 7] == -(0.5 + 0.25) and pen[1, 9] == -1.0


def test_empty_lists_are_noop():
    a, info = _orch()
    a.cumulate_output_token_lists([[], []])
    assert not np.any(a.apply())
