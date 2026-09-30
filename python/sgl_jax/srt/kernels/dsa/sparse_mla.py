"""Sparse absorbed-MLA via the existing dense v2 Pallas kernel.

Two entry points:

- ``sparse_mla_page_level`` (preferred, decode T=1): re-indexes
  ``page_indices`` to only the pages touched by ``topk_indices`` (plus the
  new-token page appended last so the kernel's in-place kv write lands at
  the true slot), then runs the unmodified dense kernel. No kv gather, no
  temp buffer — the kernel DMAs ~k/page_size pages straight from the real
  cache instead of all L/page_size. Page-level granularity (attends the
  whole page, a superset of the token-level topk) so correctness is
  preserved.

- ``sparse_mla_via_dense`` (fallback / T>1 reference): gathers the k
  selected kv rows into a temp page buffer and runs the dense kernel over
  it with kv_len=k. Exact token-level sparse but pays a [T,k,D] HBM gather
  per layer, which dominates at T=1 (0.47ms vs the 0.1ms dense@2K it
  wraps).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.mla.v2.kernel import (
    get_kv_cache_shape,
    mla_ragged_paged_attention,
)


@functools.partial(jax.jit, static_argnames=("page_size", "pages_per_seq", "k_pages_max"))
def compute_topk_pages(
    topk_indices: jax.Array,
    *,
    page_size: int,
    pages_per_seq: int,
    k_pages_max: int,
) -> jax.Array:
    """Unique seq-local pages touched by ``topk_indices``, padded to k_pages_max.

    Returns i32[T, k_pages_max] with ascending page ids and -1 padding.
    ``topk_indices`` rows are in indexer order: descending score, -1 padding.
    When more than k_pages_max pages are touched, each page is ranked by its
    best (first-occurring) token and the k_pages_max best-ranked pages win.
    Sort only the selected token slots, avoiding a one-hot membership tensor
    whose logical size is T * selected_tokens * pages_per_seq.
    """
    pages = topk_indices // page_size
    valid = (topk_indices >= 0) & (pages < pages_per_seq)
    # The sentinel sorts after every valid page so invalid entries drop out.
    pages = jnp.where(valid, pages, pages_per_seq)
    rank = jnp.broadcast_to(jnp.arange(pages.shape[-1], dtype=jnp.int32), pages.shape)
    # Stable sort by page keeps each page's best-ranked slot first in its run.
    pages, rank = jax.lax.sort((pages, rank), dimension=-1, num_keys=1, is_stable=True)
    first = jnp.ones_like(pages[:, :1], dtype=jnp.bool_)
    unique = jnp.concatenate([first, pages[:, 1:] != pages[:, :-1]], axis=-1)
    unique &= pages < pages_per_seq
    pages = jnp.where(unique, pages, pages_per_seq)
    rank = jnp.where(unique, rank, jnp.iinfo(jnp.int32).max)

    # Keep the best-ranked unique pages, then restore ascending page order.
    k_eff = min(k_pages_max, pages.shape[-1])
    _, keep = jax.lax.top_k(-rank, k_eff)
    hit_pages = jnp.sort(jnp.take_along_axis(pages, keep, axis=-1), axis=-1)
    hit_pages = jnp.where(hit_pages < pages_per_seq, hit_pages, -1)
    return jnp.pad(hit_pages, ((0, 0), (0, k_pages_max - k_eff)), constant_values=-1)


def page_mask_to_list(mask: jax.Array, width: int) -> tuple[jax.Array, jax.Array]:
    """``[N, P]`` hit mask -> (lowest ``width`` hit pages ascending ``[N, width]`` (-1 padded),
    their count ``[N]``).

    A ``P``-wide sort of page ids (``P`` = pages per sequence) replaces sorting the
    selected tokens/pages themselves.
    """
    P = mask.shape[-1]
    big = jnp.iinfo(jnp.int32).max
    ids = jnp.broadcast_to(jnp.arange(P, dtype=jnp.int32), mask.shape)
    hits = jnp.sort(jnp.where(mask, ids, big), axis=-1)
    hits = jnp.where(hits < big, hits, -1)
    if width > P:
        hits = jnp.pad(hits, ((0, 0), (0, width - P)), constant_values=-1)
    return hits[:, :width], jnp.minimum(jnp.sum(mask, axis=-1), width).astype(jnp.int32)


def pages_to_mask(page_ids: jax.Array, num_pages: int) -> jax.Array:
    """``[N, M]`` page ids (-1 = none) -> ``[N, num_pages]`` hit mask (dense compare, no sort)."""
    ar = jnp.arange(num_pages, dtype=page_ids.dtype)
    return jnp.any(page_ids[:, :, None] == ar[None, None, :], axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "page_size",
        "pages_per_seq",
        "kv_lora_rank",
        "k_pages_max",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "page_share_group",
        "group_queries",
        "mask_pages",
    ),
)
def sparse_mla_page_level(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    topk_pages: jax.Array | None = None,
    *,
    sm_scale: float,
    page_size: int,
    pages_per_seq: int,
    kv_lora_rank: int,
    k_pages_max: int = 64,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int | None = None,
    page_share_group: int = 1,
    group_queries: bool = False,
    mask_pages: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Page-level sparse MLA: dense kernel over only the topk-touched pages.

    ``page_share_group`` > 1 (opt-in A/B for decode-form speculative verify,
    where every group of ``page_share_group`` consecutive tokens belongs to one
    request): all tokens of a group use the LAST token's page list, so the
    ``page_indices[start + local]`` physical-page gather runs once per group
    instead of once per token. Changes the attention page set of the earlier
    tokens in the group; not for default use.
    ``group_queries`` (with ``page_share_group`` = G > 1): the G tokens of a
    request become ONE ragged sequence with G query rows instead of G
    one-query pseudo-sequences, so the kernel reads the request's KV once and
    applies its causal rule (query i sees k_pos <= kv_len - G + i). Layout:
    [last token's hit pages (minus the previous page when the group crosses a
    page boundary), previous page if crossing, new page, pad]; the tail G
    positions are then exactly the G consecutive tokens, so the in-place
    new-kv writes land on the true slots. Page set = the last token's
    selection (identical to per-token when every page is selected).

    Decode-only (each seq contributes exactly one query token). Builds a
    per-seq page list of length ``k_pages_max``:
      [unique pages of topk_indices ... pad ... new-token's page]
    and hands that as ``page_indices`` to the unmodified dense kernel with
    ``kv_len`` set so the kernel's in-place new-kv write lands on the true
    slot (last page × page_size + new_off + 1 - 1).

    Returns ``(o_latent, updated_cache_kv)`` — the cache is written in place
    by the dense kernel, so no separate ``_scatter_paged`` is needed.
    """
    T = ql_nope.shape[0]
    S = kv_lens.shape[0]

    # Per-token seq id (decode: token t belongs to seq t after cu_q_lens=[0..T]).
    seq_id = jnp.searchsorted(cu_q_lens[1:], jnp.arange(T), side="right")
    seq_id = jnp.clip(seq_id, 0, S - 1)
    tok_valid = kv_lens[seq_id] > 0  # [T] — padding seqs (bs padding) have kv_len==0
    # page_indices is packed per-seq at cu_kv_lens[i]//page_size (variable stride
    # via cumsum(aligned_lens)), NOT seq_id*pages_per_seq.
    seq_page_start = cu_kv_lens[seq_id] // page_size  # [T]

    valid = topk_indices >= 0
    safe_idx = jnp.where(valid, topk_indices, 0)
    page_local = safe_idx // page_size  # [T, k] seq-local page ids

    # New token's seq-local slot = kv_len-1; its page/offset. Clamp for padding
    # seqs (kv_len==0) so the -1 doesn't index page_indices[-1] and corrupt the
    # previous real seq's tail page via the kernel's in-place kv write.
    new_abs = jnp.maximum(kv_lens[seq_id] - 1, 0)  # [T]
    new_page_local = new_abs // page_size
    new_off = new_abs % page_size

    def hit_list(page_ids, num_pages):
        # Ascending unique pages of page_ids (-1 = none) minus the new-token page, which
        # the sp_local layout below re-appends after hit_pages[:n_hit_c].
        mask = pages_to_mask(page_ids, num_pages)
        mask &= jnp.arange(num_pages)[None, :] != new_page_local[:, None]
        return page_mask_to_list(mask, k_pages_max - 1)

    if topk_pages is not None:
        # Precomputed by the indexer (shared across the 4-layer IndexShare
        # group), so this branch skips the per-layer page mask over all k
        # selected tokens. Valid entries are still compacted to the front —
        # new_page may sit in the middle of topk_pages (compute_topk_pages
        # doesn't know it), and the sp_local layout below assumes
        # hit_pages[:n_hit_c] are all valid.
        raw = topk_pages[..., : k_pages_max - 1]
        mp = pages_per_seq if mask_pages is None else min(mask_pages, pages_per_seq)

        def _sorted():
            hit_is_valid = (raw >= 0) & (raw != new_page_local[:, None])
            n = jnp.minimum(jnp.sum(hit_is_valid, -1), k_pages_max - 1)
            h = jnp.sort(jnp.where(hit_is_valid, raw, jnp.iinfo(jnp.int32).max), axis=-1)
            return h, n.astype(jnp.int32)

        if mp <= k_pages_max - 1:
            # Hit mask over the sequence's pages + an mp-wide compaction instead of
            # sorting the (k_pages_max-1)-wide page list. Runtime guard: a sequence
            # longer than mp pages takes the sort path.
            fits = jnp.max((kv_lens + page_size - 1) // page_size) <= mp
            hit_pages, n_hit_c = jax.lax.cond(fits, lambda: hit_list(raw, mp), _sorted)
        else:
            hit_pages, n_hit_c = _sorted()
    else:
        # Unique pages via a page mask over the k topk positions (page count is
        # bounded by pages_per_seq which is static). Keeps everything jit-static.
        hit_pages, n_hit_c = hit_list(jnp.where(valid, page_local, -1), pages_per_seq)
    n_used = n_hit_c + 1  # +1 for the new-token page

    # Layout: [hit_pages[0..n_hit_c-1], new_page, pad...] so the kernel's
    # kv_len mask (k_span < sp_kv_len) drops the pad tail while keeping the
    # new-token page inside. Padding points at the sentinel last cache page.
    col = jnp.arange(k_pages_max)[None, :]  # [1, kp]
    sp_local = jnp.where(
        col < n_hit_c[:, None],
        jnp.pad(hit_pages, ((0, 0), (0, 1))),
        jnp.where(col == n_hit_c[:, None], new_page_local[:, None], 0),
    )
    grouped = bool(group_queries) and page_share_group > 1 and T % page_share_group == 0
    if grouped:
        G = page_share_group
        Ng = T // G
        rep = slice(G - 1, None, G)  # last token of every group
        int_max = jnp.iinfo(jnp.int32).max
        new_pl = new_page_local[rep]  # [Ng]
        new_of = new_off[rep]
        # the group's G consecutive positions end at new_of inside new_page; if
        # fewer than G slots precede it there, the group starts in the previous page
        straddle = new_of < (G - 1)
        prev_pl = jnp.maximum(new_pl - 1, 0)
        hp = hit_pages[rep]  # [Ng, kp-1], valid entries first
        colh = jnp.arange(k_pages_max - 1)[None, :]
        hv = colh < n_hit_c[rep][:, None]
        is_prev = hv & straddle[:, None] & (hp == prev_pl[:, None])
        hp2 = jnp.sort(jnp.where(hv & ~is_prev, hp, int_max), axis=-1)
        s_i = straddle.astype(jnp.int32)
        n_hit_g = n_hit_c[rep] - jnp.sum(is_prev, axis=-1)
        n_hit_g = jnp.minimum(n_hit_g, k_pages_max - 1 - s_i)
        n_used_g = n_hit_g + 1 + s_i
        colg = jnp.arange(k_pages_max)[None, :]
        hp2p = jnp.pad(hp2, ((0, 0), (0, 1)))
        tail0 = jnp.where(straddle, prev_pl, new_pl)
        sp_local_g = jnp.where(
            colg < n_hit_g[:, None],
            hp2p,
            jnp.where(
                colg == n_hit_g[:, None],
                tail0[:, None],
                jnp.where((colg == n_hit_g[:, None] + 1) & straddle[:, None], new_pl[:, None], 0),
            ),
        )
        valid_g = tok_valid[rep]
        sp_phys = jnp.where(
            (colg < n_used_g[:, None]) & valid_g[:, None],
            page_indices[seq_page_start[rep][:, None] + sp_local_g],
            cache_kv.shape[0] - 1,
        )
        # bs-padding groups (kv_len == 0) still carry G query rows, and the mixed
        # kernel sizes its KV loop by kv_len: kv_len == 0 skips the loop, so the
        # query-block DMA is never waited (semaphore non-zero at kernel exit) or
        # the next sequence is never prefetched (hang). Give them G all-new
        # positions on the sentinel page, like the per-token path's kv_len == 1.
        sp_kv_len = jnp.where(
            valid_g, n_hit_g * page_size + s_i * page_size + new_of + 1, G
        ).astype(jnp.int32)
        sp_page_indices = sp_phys.reshape(-1)  # [Ng * k_pages_max]
        sp_cu_q = jnp.arange(Ng + 1, dtype=jnp.int32) * G
        sp_cu_kv = jnp.arange(Ng + 1, dtype=jnp.int32) * (k_pages_max * page_size)
        sp_dist = jnp.array([0, 0, Ng], dtype=jnp.int32)  # all sequences are mixed
    elif page_share_group > 1 and T % page_share_group == 0:
        G = page_share_group
        rep = slice(G - 1, None, G)  # last token of every group
        sp_local_g = sp_local[rep]
        phys_g = page_indices[seq_page_start[rep][:, None] + sp_local_g]  # [T/G, kp]
        sp_phys = jnp.where(
            (col < jnp.repeat(n_used[rep], G)[:, None]) & tok_valid[:, None],
            jnp.repeat(phys_g, G, axis=0),
            cache_kv.shape[0] - 1,
        )
        n_hit_c = jnp.repeat(n_hit_c[rep], G)
    else:
        sp_phys = jnp.where(
            (col < n_used[:, None]) & tok_valid[:, None],
            page_indices[seq_page_start[:, None] + sp_local],
            cache_kv.shape[0] - 1,
        )

    if not grouped:
        # Metadata for the dense kernel: T "sequences", each kv_len positions.
        sp_kv_len = (n_hit_c * page_size + new_off + 1).astype(jnp.int32)  # [T]
        sp_page_indices = sp_phys.reshape(-1)  # [T * k_pages_max]
        sp_cu_q = jnp.arange(T + 1, dtype=jnp.int32)
        sp_cu_kv = jnp.arange(T + 1, dtype=jnp.int32) * (k_pages_max * page_size)
        sp_dist = jnp.array([T, T, T], dtype=jnp.int32)

    o, cache_out = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        sp_kv_len,
        sp_page_indices,
        sp_cu_q,
        sp_cu_kv,
        sp_dist,
        sm_scale=sm_scale,
        sliding_window=None,
        soft_cap=None,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        # grouped: every pseudo-sequence has exactly G query rows.
        mixed_static_q_len=page_share_group if grouped else None,
    )
    return o[..., :kv_lora_rank], cache_out


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "page_size",
        "queries_per_seq",
        "kv_lora_rank",
        "k_pages_max",
        "vmem_limit_bytes",
    ),
)
def sparse_mla_chain_request_level(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    topk_pages: jax.Array,
    req_seq_lens: jax.Array,
    req_page_indices: jax.Array,
    req_distribution: jax.Array,
    *,
    sm_scale: float,
    page_size: int,
    queries_per_seq: int,
    kv_lora_rank: int,
    k_pages_max: int = 512,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Page-level sparse MLA for speculative chains, one causal ``Tq``-query pass per request.

    ``sparse_mla_page_level`` runs each of a request's ``Tq`` chain tokens as its own
    one-query sequence, so the request's hit pages are read ``Tq`` times. Here each request
    attends the union of its queries' hit pages once, with the chain's own page(s) appended
    last so the chain tokens are the final ``Tq`` positions and the ragged kernel's causal
    rule (query ``i`` sees ``kv_len - Tq + i`` and earlier) applies. NoPE MLA is position
    free, so page order is irrelevant. Same page-level (superset) semantics as the per-query
    path; each query may additionally see pages selected by its siblings.

    ``topk_pages`` [B*Tq, k] seq-local hit pages per chain token (-1 padded, request-major);
    ``req_seq_lens`` [B] = committed + Tq; ``req_page_indices`` fixed stride [B * pps].
    Returns ``(o_latent [B*Tq, H, kv_lora_rank], updated cache)``.
    """
    Tq = queries_per_seq
    B = req_seq_lens.shape[0]
    pps = req_page_indices.shape[0] // B
    big = jnp.iinfo(jnp.int32).max

    last = jnp.maximum(req_seq_lens - 1, 0)
    first_chain_page = jnp.maximum(req_seq_lens - Tq, 0) // page_size
    last_chain_page = last // page_size
    n_chain_pages = last_chain_page - first_chain_page + 1  # 1 or 2

    hits = topk_pages.reshape(B, -1)
    ok = (hits >= 0) & (hits < first_chain_page[:, None])
    hits = jnp.sort(jnp.where(ok, hits, big), axis=-1)
    uniq = jnp.concatenate(
        [hits[:, :1] < big, (hits[:, 1:] != hits[:, :-1]) & (hits[:, 1:] < big)], axis=-1
    )
    # Compact the unique pages to the front (ascending); cap leaves room for the chain pages.
    n_hit = jnp.minimum(jnp.sum(uniq, axis=-1), k_pages_max - 2)
    hit_pages = jnp.sort(jnp.where(uniq, hits, big), axis=-1)[:, : k_pages_max - 2]

    col = jnp.arange(k_pages_max)[None, :]
    chain_col = col - n_hit[:, None]  # 0 / 1 -> first / second chain page
    local = jnp.where(
        col < n_hit[:, None],
        jnp.pad(hit_pages, ((0, 0), (0, 2)), constant_values=0),
        first_chain_page[:, None] + chain_col,
    )
    used = col < (n_hit + n_chain_pages)[:, None]
    active = (req_seq_lens > 0)[:, None]
    local = jnp.clip(local, 0, pps - 1)
    phys = jnp.where(
        used & active,
        req_page_indices[jnp.arange(B)[:, None] * pps + local],
        cache_kv.shape[0] - 1,
    )
    sp_kv_len = jnp.where(
        req_seq_lens > 0,
        n_hit * page_size + (last - first_chain_page * page_size) + 1,
        0,
    ).astype(jnp.int32)
    sp_cu_q = jnp.arange(B + 1, dtype=jnp.int32) * Tq
    sp_cu_kv = jnp.arange(B + 1, dtype=jnp.int32) * (k_pages_max * page_size)

    o, cache_out = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        sp_kv_len,
        phys.reshape(-1),
        sp_cu_q,
        sp_cu_kv,
        req_distribution,
        sm_scale=sm_scale,
        sliding_window=None,
        soft_cap=None,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return o[..., :kv_lora_rank], cache_out


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "page_size",
        "pages_per_seq",
        "kv_lora_rank",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def sparse_mla_via_dense(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    page_size: int,
    pages_per_seq: int,
    kv_lora_rank: int,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """Absorbed-MLA over per-query top-k, delegating to the dense v2 kernel.

    Args mirror ``mla_ragged_paged_attention`` plus:
      topk_indices: i32[T, k]  per-query KV positions (seq-relative, -1 = pad)
      pages_per_seq: static stride into ``page_indices``

    Returns:
      o_latent: f[T, H, kv_lora_rank]
    """
    T, H, _ = ql_nope.shape
    k = topk_indices.shape[1]
    _, ps_over_pack, pack, kv_dim = cache_kv.shape
    assert ps_over_pack * pack == page_size
    S = kv_lens.shape[0]

    # ── 1. resolve topk seq-relative positions → flat physical slot ──────
    seq_id = jnp.searchsorted(cu_q_lens[1:], jnp.arange(T), side="right")
    seq_id = jnp.clip(seq_id, 0, S - 1)
    seq_page_start = cu_kv_lens[seq_id] // page_size
    valid = topk_indices >= 0
    safe_idx = jnp.where(valid, topk_indices, 0)
    page_local = safe_idx // page_size
    off = safe_idx % page_size
    phys_page = page_indices[seq_page_start[:, None] + page_local]
    slot = phys_page * page_size + off  # [T, k]

    # ── 2. gather kv rows via 1-D index (2-3× faster than 3-D fancy index
    # on TPU) → contiguous [T, k, kv_dim] → temp pages ────────────────────
    cache_flat = cache_kv.reshape(-1, kv_dim)
    kv_gathered = jnp.take(cache_flat, slot.reshape(-1), axis=0).reshape(T, k, kv_dim)
    kv_gathered = jnp.where(valid[..., None], kv_gathered, 0)

    n_tmp_pages = (k + page_size - 1) // page_size
    pad = n_tmp_pages * page_size - k
    kv_padded = jnp.pad(kv_gathered, ((0, 0), (0, pad), (0, 0)))
    tmp_shape = get_kv_cache_shape(T * n_tmp_pages + 1, page_size, kv_dim, cache_kv.dtype)
    tmp_cache = kv_padded.reshape(T * n_tmp_pages, ps_over_pack, pack, kv_dim)
    tmp_cache = jnp.concatenate(
        [tmp_cache, jnp.zeros((tmp_shape[0] - T * n_tmp_pages, *tmp_shape[1:]), cache_kv.dtype)],
        axis=0,
    )

    # ── 3. build metadata: each token is its own "sequence" of kv_len=k ──
    n_valid = jnp.sum(valid, axis=-1).astype(jnp.int32)
    tmp_kv_lens = n_valid  # i32[T]
    tmp_page_indices = jnp.arange(T * n_tmp_pages, dtype=jnp.int32)
    tmp_cu_q = jnp.arange(T + 1, dtype=jnp.int32)
    tmp_cu_kv = jnp.arange(T + 1, dtype=jnp.int32) * (n_tmp_pages * page_size)
    tmp_dist = jnp.array([T, T, T], dtype=jnp.int32)

    # ── 4. run dense kernel over the gathered buffer (O(k) not O(L)) ─────
    o_latent, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        tmp_cache,
        tmp_kv_lens,
        tmp_page_indices,
        tmp_cu_q,
        tmp_cu_kv,
        tmp_dist,
        sm_scale=sm_scale,
        sliding_window=None,
        soft_cap=None,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return o_latent[..., :kv_lora_rank]
