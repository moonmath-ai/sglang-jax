"""CPU-side tests for live indexer row grouping and output restoration."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa import streamindex_live

PAGE_SIZE, HEAD_DIM, HEADS, TOPK, CAPACITY = 128, 128, 32, 8, 65


def fake_streamindex_topk(q, weights, cache, seq_lens, pages, cu_q, distribution, *, k, **kwargs):
    """Returns each row's first page id, then the number of rows scored in the call."""
    del q, weights, cache, cu_q, distribution, kwargs
    rows = seq_lens.shape[0]
    first_pages = pages.reshape(rows, -1)[:, 0]
    out = jnp.broadcast_to(first_pages[:, None], (rows, k))
    return out.at[:, 1].set(rows)


@pytest.fixture
def fake_scorer(monkeypatch):
    # The scorer is looked up at trace time: drop cached traces on both sides of the test.
    jax.clear_caches()
    monkeypatch.setattr(streamindex_live, "streamindex_topk", fake_streamindex_topk)
    yield
    jax.clear_caches()


@pytest.mark.parametrize(
    "lengths, active, rows_per_call",
    [
        # > 4 active rows: one pass over all rows, padded to blocks of 4.
        ([100, 200, 300, 2049, 2050, 4097, 5000, 8193, 129, 0], 10, 12),
        # <= 4 active rows: same-bucket groups of 4 in a larger batch.
        ([100, 2049, 0, 8193, 4097, 5000, 300, 200], 5, 4),
        # batch <= 4: only the grouped path is traced.
        ([2049, 100, 8193], 3, 3),
    ],
)
def test_rows_restore_and_packed_page_offsets(fake_scorer, lengths, active, rows_per_call):
    lengths = np.asarray(lengths, np.int32)
    batch = len(lengths)
    page_counts = (lengths + PAGE_SIZE - 1) // PAGE_SIZE
    offsets = np.r_[0, np.cumsum(page_counts, dtype=np.int32)]

    # Encode request and page number in each packed page id. The fake scorer
    # returns each row's first page id, checking the repacked offsets and that
    # the results are scattered back to their original rows.
    page_indices = np.full(batch * CAPACITY, -1, np.int32)
    for row, count in enumerate(page_counts):
        start = offsets[row]
        page_indices[start : start + count] = row * 100 + np.arange(count) + 1

    q = np.zeros((batch, HEADS, HEAD_DIM), ml_dtypes.bfloat16)
    weights = np.zeros((batch, HEADS), ml_dtypes.bfloat16)
    cache = np.zeros((1, PAGE_SIZE, HEAD_DIM), ml_dtypes.bfloat16)
    cu_q_lens = np.minimum(np.arange(batch + 1, dtype=np.int32), active)
    cu_kv_lens = offsets * PAGE_SIZE
    distribution = np.asarray([active] * 3, np.int32)

    inputs = map(
        jnp.asarray,
        (q, weights, cache, lengths, page_indices, cu_q_lens, cu_kv_lens, distribution),
    )
    result = np.asarray(
        streamindex_live.streamindex_topk_live(*inputs, k=TOPK, pages_per_seq=CAPACITY)
    )

    live = (np.arange(batch) < active) & (lengths > 0)
    expected = np.repeat((np.arange(batch, dtype=np.int32) * 100 + 1)[:, None], TOPK, axis=1)
    expected[:, 1] = rows_per_call
    expected[~live] = -1
    np.testing.assert_array_equal(result, expected)
