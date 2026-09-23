"""Page deduplication must preserve page order, truncation and invalid slots."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.sparse_mla import compute_topk_pages


def _expected(tokens, page_size, pages_per_seq, budget):
    result = np.full((len(tokens), budget), -1, dtype=np.int32)
    for row, ids in enumerate(tokens):
        valid = ids[(ids >= 0) & (ids // page_size < pages_per_seq)]
        pages = np.unique(valid // page_size)[:budget]
        result[row, : len(pages)] = pages
    return result


@pytest.mark.parametrize(
    "batch,selected,page_size,pages_per_seq,budget",
    [
        (1, 16, 128, 8, 4),
        (4, 32, 7, 17, 8),
        (2, 8, 128, 3, 16),
        (4, 2048, 128, 7635, 512),
        (2, 0, 128, 4, 8),
        (2, 8, 128, 4, 0),
    ],
)
@pytest.mark.parametrize("pattern", ["random", "invalid", "duplicates"])
def test_page_selection(batch, selected, page_size, pages_per_seq, budget, pattern):
    rng = np.random.default_rng(42)
    if pattern == "invalid":
        tokens = np.full((batch, selected), -1, dtype=np.int32)
    elif pattern == "duplicates":
        tokens = rng.integers(
            0, min(pages_per_seq, 3) * page_size, (batch, selected), dtype=np.int32
        )
    else:
        tokens = rng.integers(
            -page_size, (pages_per_seq + 2) * page_size, (batch, selected), dtype=np.int32
        )
    actual = compute_topk_pages(
        jnp.asarray(tokens),
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        k_pages_max=budget,
    )
    np.testing.assert_array_equal(actual, _expected(tokens, page_size, pages_per_seq, budget))


def test_overflow_keeps_lowest_pages_instead_of_token_rank():
    tokens = (np.arange(2048, dtype=np.int32)[::-1] * 128)[None, :]
    actual = compute_topk_pages(
        jnp.asarray(tokens), page_size=128, pages_per_seq=7635, k_pages_max=512
    )
    np.testing.assert_array_equal(actual, np.arange(512, dtype=np.int32)[None, :])


def test_page_boundaries_and_out_of_range_tokens():
    tokens = np.array([[127, 128, 129, 255, 256, 511, 512, -1, -128]], dtype=np.int32)
    actual = compute_topk_pages(
        jnp.asarray(tokens), page_size=128, pages_per_seq=4, k_pages_max=6
    )
    np.testing.assert_array_equal(actual, [[0, 1, 2, 3, -1, -1]])
