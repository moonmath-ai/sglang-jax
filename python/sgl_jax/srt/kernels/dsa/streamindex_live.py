"""Decode indexer with per-request live-context dispatch.

Pallas reads only live KV blocks. A runtime switch chooses a score-buffer
bucket per request, so the exit-stage top-k also avoids the global capacity.
This is independent of the packed page-table stride and preserves the full
context limit. No short-context shortcut is used: every active query is scored.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk


def _page_buckets(pages_per_seq: int, pages_per_block: int) -> tuple[int, ...]:
    """Powers of two blocks, capped at the caller's actual page-table capacity."""
    if pages_per_seq < 1 or pages_per_block < 1:
        raise ValueError("Page capacity and block size must be positive")
    buckets = []
    pages = pages_per_block
    while pages < pages_per_seq:
        buckets.append(pages)
        pages *= 2
    buckets.append(pages_per_seq)
    return tuple(buckets)


@functools.partial(
    jax.jit,
    static_argnames=("k", "pages_per_seq", "num_kv_pages_per_block"),
)
def streamindex_topk_live(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    num_kv_pages_per_block: int = 64,
) -> jax.Array:
    """Return int32[B,k] seq-local token indices, with trailing -1 padding.

    Decode ABI matches ``streamindex_topk_ref(one_token_per_seq=True)``:
    one query per sequence, packed page table, cu_kv_lens in TOKEN units,
    and an uncompressed BF16 cache [physical_pages,page_size,head_dim].
    The caller writes current keys before invoking this function.

    Each active row selects its own bucket at runtime. Do not vmap the switch:
    that would evaluate every bucket and restore full-capacity work. Requests
    currently execute sequentially; independent request scheduling is separate
    from eliminating capacity-dependent work.

    Scoring follows the existing Pallas kernel; selection is exact top-k
    (subject to floating-point scoring and tie ordering), unlike the reference's
    approximate candidate reduction. Tests compare against an exact score oracle.
    Oversized rows use exact chunk-local top-k followed by a merge, avoiding
    a full-row sort when the score bucket exceeds SparseCore VMEM.
    Live sequence lengths must not exceed pages_per_seq * page_size.
    """
    del cu_q_lens  # In decode, query i belongs to sequence i.
    batch = q.shape[0]
    if q.ndim != 3 or batch != seq_lens.shape[0]:
        raise ValueError("Live indexer requires one query per sequence")
    if cache_kv.ndim != 3 or q.shape[-1] != cache_kv.shape[-1]:
        raise ValueError("Expected [pages,page_size,head_dim] indexer cache")
    if q.shape[-1] % 128:
        raise ValueError("Pallas indexer requires a head dimension divisible by 128")
    if k < 1:
        raise ValueError("k must be positive")
    page_size = cache_kv.shape[1]
    if page_indices.shape[0] < pages_per_seq:
        raise ValueError("Page table is smaller than pages_per_seq")
    if page_size * num_kv_pages_per_block % 128:
        raise ValueError("KV block must contain a multiple of 128 tokens")
    # Keep 2K-token buckets at page_size=128, while larger contexts amortize
    # paged-DMA/control overhead over up to 64 pages per scoring block.
    buckets = _page_buckets(pages_per_seq, min(16, num_kv_pages_per_block))
    limits = jnp.asarray([pages * page_size for pages in buckets], jnp.int32)
    # A view of the existing cache, not a gathered copy. Packing is part of the
    # Pallas cache ABI: two BF16 elements per 32-bit word.
    if cache_kv.dtype != jnp.bfloat16 or page_size % 2:
        raise ValueError("Live indexer requires BF16 cache and an even page size")
    cache4d = cache_kv.reshape(cache_kv.shape[0], page_size // 2, 2, cache_kv.shape[-1])
    one_cuq = jnp.asarray([0, 1], jnp.int32)
    one_dist = jnp.asarray([1, 1, 1], jnp.int32)

    def make_branch(bucket_pages):
        def score_and_select(args):
            q_row, w_row, length, page_start = args
            pages = jax.lax.dynamic_slice_in_dim(page_indices, page_start, bucket_pages)
            return streamindex_topk(
                q_row,
                w_row,
                cache4d,
                length,
                pages,
                one_cuq,
                one_dist,
                k=k,
                compression_ratio=1,
                num_kv_pages_per_block=min(bucket_pages, num_kv_pages_per_block),
                num_queries_per_block=1,
                decode_req_batch_size=1,
                topk_backend="chunked",
            )

        return score_and_select

    branches = tuple(make_branch(pages) for pages in buckets)

    def row_step(seq_id, out):
        length = jax.lax.dynamic_slice_in_dim(seq_lens, seq_id, 1)
        active = (seq_id < distribution[0]) & (length[0] > 0)
        q_row = jax.lax.dynamic_slice_in_dim(q, seq_id, 1)
        w_row = jax.lax.dynamic_slice_in_dim(weights, seq_id, 1)
        page_start = cu_kv_lens[seq_id] // page_size
        bucket = jnp.sum(length[0] > limits, dtype=jnp.int32)
        indices = jax.lax.cond(
            active,
            lambda args: jax.lax.switch(bucket, branches, args),
            lambda args: jnp.full((1, k), -1, jnp.int32),
            (q_row, w_row, length, page_start),
        )
        return jax.lax.dynamic_update_slice_in_dim(out, indices, seq_id, 0)

    return jax.lax.fori_loop(0, batch, row_step, jnp.full((batch, k), -1, jnp.int32))
