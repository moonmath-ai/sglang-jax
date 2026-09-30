"""Decode indexer with per-request live-context dispatch.

Pallas reads only live KV blocks. A runtime switch chooses a score-buffer
bucket from the live context, so the exit-stage top-k also avoids the global
capacity. This is independent of the packed page-table stride and preserves the
full context limit. No short-context shortcut is used: every active query is scored.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk

# Active rows above which one pass at the batch's longest live bucket replaces the grouped
# loop (one row group of the grouped loop).
_SINGLE_PASS_MIN_ROWS = 4


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


@jax.jit(static_argnames=("k", "pages_per_seq", "num_kv_pages_per_block", "compression_ratio"))
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
    compression_ratio: int = 1,
) -> jax.Array:
    """Return int32[B,k] seq-local token indices, with trailing -1 padding.

    ``compression_ratio > 1`` (KPool): the cache holds one pooled entry per ``ratio``
    tokens (``page_size`` entries per page = ``page_size * ratio`` tokens), seq_lens and
    cu_kv_lens stay in token units, and the result is in pooled-entry space.

    Decode ABI matches ``streamindex_topk_ref(one_token_per_seq=True)``:
    one query per sequence, packed page table, cu_kv_lens in TOKEN units,
    and an uncompressed BF16 cache [physical_pages,page_size,head_dim].
    The caller writes current keys before invoking this function.

    Rows are bucketed by their live context (powers of two pages), so work and the exit-stage
    top-k stay bounded by the live context rather than the page-table capacity: either one
    pass at the batch's longest bucket, or (few active rows) same-bucket groups of up to
    four. Do not vmap the bucket switch: that would evaluate every bucket and restore
    full-capacity work.

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
    limits = jnp.asarray([pages * page_size * compression_ratio for pages in buckets], jnp.int32)
    # A view of the existing cache, not a gathered copy. Packing is part of the
    # Pallas cache ABI: two BF16 elements per 32-bit word.
    if cache_kv.dtype != jnp.bfloat16 or page_size % 2:
        raise ValueError("Live indexer requires BF16 cache and an even page size")
    cache4d = cache_kv.reshape(cache_kv.shape[0], page_size // 2, 2, cache_kv.shape[-1])
    if batch == 0:
        return jnp.full((0, k), -1, jnp.int32)

    live = (jnp.arange(batch) < distribution[0]) & (seq_lens > 0)
    lengths = jnp.where(live, seq_lens, 0)
    # Pad by the largest window so every row window stays in bounds without shifting.
    starts = cu_kv_lens[:batch] // (page_size * compression_ratio)
    padded_pages = jnp.concatenate(
        [page_indices, jnp.full((buckets[-1],), page_indices[-1], page_indices.dtype)]
    )

    def windows(rows, bp):
        row_starts = starts if rows is None else starts[rows]
        return padded_pages[row_starts[:, None] + jnp.arange(bp, dtype=jnp.int32)].reshape(-1)

    def score(q_rows, w_rows, lengths_rows, pages, bp):
        n = lengths_rows.shape[0]
        return streamindex_topk(
            q_rows,
            w_rows,
            cache4d,
            lengths_rows,
            pages,
            jnp.arange(n + 1, dtype=jnp.int32),
            jnp.asarray([n, n, n], jnp.int32),
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=min(bp, num_kv_pages_per_block),
            num_queries_per_block=1,
            decode_req_batch_size=min(4, n),
        )

    def _single_pass():
        # Decode rows go to the kernel in blocks of 4 (only traced for batch > 4).
        padded = -(-batch // 4) * 4
        pad = padded - batch
        rows = None if pad == 0 else jnp.minimum(jnp.arange(padded, dtype=jnp.int32), batch - 1)
        qp = jnp.pad(q, ((0, pad), (0, 0), (0, 0)))
        wp = jnp.pad(weights, ((0, pad), (0, 0)))
        lengths_p = jnp.pad(lengths, ((0, pad),))
        bucket = jnp.minimum(jnp.sum(jnp.max(lengths) > limits, dtype=jnp.int32), len(buckets) - 1)

        def _branch(bp):
            return lambda _: score(qp, wp, lengths_p, windows(rows, bp), bp)

        sel = jax.lax.switch(bucket, [_branch(bp) for bp in buckets], None)[:batch]
        return jnp.where((lengths > 0)[:, None], sel, -1)

    def _grouped():
        group_size = min(4, batch)
        # Leave one group's lookahead after the real rows: a bucket run can end
        # at any row, so the final partial group still needs a full static slice.
        padded_batch = ((batch + group_size - 1) // group_size) * group_size + group_size - 1
        pad = padded_batch - batch
        qb = jnp.pad(q, ((0, pad), (0, 0), (0, 0)))
        wb = jnp.pad(weights, ((0, pad), (0, 0)))
        lengths_b = jnp.pad(lengths, ((0, pad),))

        row_buckets = jnp.sum(lengths_b[:, None] > limits[None, :], axis=1, dtype=jnp.int32)
        row_buckets = jnp.where(lengths_b > 0, row_buckets, len(buckets))
        order = jnp.argsort(row_buckets, stable=True)
        sorted_buckets = row_buckets[order]
        active_count = jnp.sum(lengths_b > 0, dtype=jnp.int32)

        def _branch(bp):
            def score_and_select(args):
                group_order, lengths_group, out = args
                selected = score(
                    qb[group_order],
                    wb[group_order],
                    lengths_group,
                    windows(jnp.minimum(group_order, batch - 1), bp),
                    bp,
                )
                keep = (lengths_group > 0)[:, None]
                return out.at[group_order].set(jnp.where(keep, selected, out[group_order]))

            return score_and_select

        def condition(carry):
            cursor, _ = carry
            return cursor < active_count

        def group_step(carry):
            cursor, out = carry
            group_order = jax.lax.dynamic_slice_in_dim(order, cursor, group_size)
            group_buckets = jax.lax.dynamic_slice_in_dim(sorted_buckets, cursor, group_size)
            bucket = group_buckets[0]
            in_bucket = group_buckets == bucket
            lengths_group = jnp.where(in_bucket, lengths_b[group_order], 0)
            out = jax.lax.switch(
                bucket,
                [_branch(bp) for bp in buckets],
                (group_order, lengths_group, out),
            )
            return cursor + jnp.sum(in_bucket, dtype=jnp.int32), out

        out = jnp.full((padded_batch, k), -1, jnp.int32)
        _, out = jax.lax.while_loop(condition, group_step, (jnp.asarray(0, jnp.int32), out))
        return out[:batch]

    if batch <= _SINGLE_PASS_MIN_ROWS:
        return _grouped()
    # The single pass scores every (padded) decode row at the bucket of the longest live
    # context, so with few active rows the grouped loop, which scores only active rows, is
    # cheaper (e.g. one long request in a bs-16 decode bucket).
    n_active = jnp.sum((jnp.arange(batch) < distribution[0]) & (seq_lens > 0))
    return jax.lax.cond(
        n_active > _SINGLE_PASS_MIN_ROWS, _single_pass, lambda: _grouped(q, weights, seq_lens)
    )


@jax.jit(static_argnames=("k", "queries_per_seq", "pages_per_seq", "num_kv_pages_per_block"))
def streamindex_topk_live_multiq(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    *,
    k: int,
    queries_per_seq: int,
    pages_per_seq: int,
    num_kv_pages_per_block: int = 64,
) -> jax.Array:
    """Live-context top-k for sequences that each end in ``queries_per_seq`` causal queries.

    Speculative chains: sequence b holds ``seq_lens[b]`` keys and its last
    ``queries_per_seq`` positions are the queries, query j seeing keys ``< seq_len - Tq + j + 1``.
    Returns int32[B * Tq, k] seq-local token indices (-1 padded) -- the same top-k as
    ``streamindex_topk_live`` over one virtual sequence per query, but each sequence's keys
    are scored in one pass. ``page_indices`` is fixed-stride ``[B * pages_per_seq]``.

    Same live-bucket dispatch as ``streamindex_topk_live``: rows are grouped by the
    power-of-two page bucket of their live length, so work and the exit-stage top-k stay
    bounded by the live context rather than the page-table capacity.
    """
    Tq = queries_per_seq
    batch = seq_lens.shape[0]
    if q.ndim != 3 or q.shape[0] != batch * Tq:
        raise ValueError("expected q [batch * queries_per_seq, heads, dim]")
    if cache_kv.ndim != 3 or q.shape[-1] != cache_kv.shape[-1]:
        raise ValueError("Expected [pages,page_size,head_dim] indexer cache")
    if q.shape[-1] % 128:
        raise ValueError("Pallas indexer requires a head dimension divisible by 128")
    page_size = cache_kv.shape[1]
    if page_indices.shape[0] < batch * pages_per_seq:
        raise ValueError("Page table is smaller than batch * pages_per_seq")
    if page_size * num_kv_pages_per_block % 128:
        raise ValueError("KV block must contain a multiple of 128 tokens")
    if cache_kv.dtype != jnp.bfloat16 or page_size % 2:
        raise ValueError("Live indexer requires BF16 cache and an even page size")
    buckets = _page_buckets(pages_per_seq, min(16, num_kv_pages_per_block))
    limits = jnp.asarray([pages * page_size for pages in buckets], jnp.int32)
    cache4d = cache_kv.reshape(cache_kv.shape[0], page_size // 2, 2, cache_kv.shape[-1])
    if batch == 0:
        return jnp.full((0, k), -1, jnp.int32)

    q = q.reshape(batch, Tq, *q.shape[1:])
    weights = weights.reshape(batch, Tq, weights.shape[-1])
    group_size = min(4, batch)
    padded_batch = ((batch + group_size - 1) // group_size) * group_size + group_size - 1
    page_starts = jnp.arange(batch, dtype=jnp.int32) * pages_per_seq
    if padded_batch != batch:
        pad = padded_batch - batch
        q = jnp.pad(q, ((0, pad), (0, 0), (0, 0), (0, 0)))
        weights = jnp.pad(weights, ((0, pad), (0, 0), (0, 0)))
        seq_lens = jnp.pad(seq_lens, ((0, pad),))
        page_starts = jnp.pad(page_starts, ((0, pad),))

    active = seq_lens > 0
    row_buckets = jnp.sum(seq_lens[:, None] > limits[None, :], axis=1, dtype=jnp.int32)
    row_buckets = jnp.where(active, row_buckets, len(buckets))
    order = jnp.argsort(row_buckets, stable=True)
    sorted_buckets = row_buckets[order]
    active_count = jnp.sum(active, dtype=jnp.int32)

    group_cuq = jnp.arange(group_size + 1, dtype=jnp.int32) * Tq
    group_dist = jnp.asarray([0, 0, group_size], jnp.int32)  # all mixed: Tq queries per seq
    # Pad by the largest window so every row slice stays in bounds without shifting (the
    # old per-element clamp only ever affected pages past a row's live length).
    padded_pages = jnp.concatenate(
        [page_indices, jnp.full((buckets[-1],), page_indices[-1], page_indices.dtype)]
    )
    bucket_branches = []
    for bucket_pages in buckets:
        def score_and_select(args, bucket_pages=bucket_pages):
            q_group, w_group, lengths_group, starts_group, group_order, out = args
            # Contiguous per-row windows of the (padded) page table: dynamic slices are
            # cheap DMAs, while a dynamic gather of a few hundred ids costs ~20 us on TPU.
            pages = jnp.concatenate(
                [
                    jax.lax.dynamic_slice_in_dim(padded_pages, starts_group[i], bucket_pages)
                    for i in range(group_size)
                ]
            )
            selected = streamindex_topk(
                q_group.reshape(group_size * Tq, *q_group.shape[2:]),
                w_group.reshape(group_size * Tq, w_group.shape[-1]),
                cache4d,
                lengths_group,
                pages,
                group_cuq,
                group_dist,
                k=k,
                compression_ratio=1,
                num_kv_pages_per_block=min(bucket_pages, num_kv_pages_per_block),
                num_queries_per_block=Tq,
                topk_backend="chunked",
            ).reshape(group_size, Tq, k)
            old = out[group_order]
            keep = (lengths_group > 0)[:, None, None]
            return out.at[group_order].set(jnp.where(keep, selected, old))

        bucket_branches.append(score_and_select)

    def condition(carry):
        cursor, _ = carry
        return cursor < active_count

    def group_step(carry):
        cursor, out = carry
        group_order = jax.lax.dynamic_slice_in_dim(order, cursor, group_size)
        group_buckets = jax.lax.dynamic_slice_in_dim(sorted_buckets, cursor, group_size)
        bucket = group_buckets[0]
        in_bucket = group_buckets == bucket
        run_size = jnp.sum(in_bucket, dtype=jnp.int32)
        lengths_group = jnp.where(in_bucket, seq_lens[group_order], 0)
        out = jax.lax.switch(
            bucket,
            tuple(bucket_branches),
            (q[group_order], weights[group_order], lengths_group, page_starts[group_order],
             group_order, out),
        )
        return cursor + run_size, out

    out = jnp.full((padded_batch, Tq, k), -1, jnp.int32)
    _, out = jax.lax.while_loop(condition, group_step, (jnp.asarray(0, jnp.int32), out))
    return out[:batch].reshape(batch * Tq, k)
