"""Request-level chain sparse MLA vs a dense NumPy reference over the hit-page union."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.sparse_mla import sparse_mla_chain_request_level
from sgl_jax.srt.kernels.mla.v2.kernel import get_kv_cache_shape

if jax.default_backend() != "tpu":
    pytest.skip("MLA v2 kernel test needs a TPU", allow_module_level=True)

PS, H, R, TQ = 16, 8, 512, 4


def _reference(ql, kv_new, cache_tok, hits, req_seq, req_pi, pps, scale):
    """ql [B*Tq, H, R]; cache_tok [pages*PS, D] token-major; hits [B*Tq, k] seq-local pages."""
    B = req_seq.shape[0]
    out = np.zeros(ql.shape, np.float32)
    for b in range(B):
        L = int(req_seq[b])
        first_chain_page = (L - TQ) // PS
        pages = sorted(
            {int(p) for p in hits[b * TQ : (b + 1) * TQ].ravel() if 0 <= p < first_chain_page}
        )
        tok = [req_pi[b * pps + p] * PS + j for p in pages for j in range(PS)]
        # committed tokens on the chain page(s) before the chain itself
        tok += [
            req_pi[b * pps + pos // PS] * PS + pos % PS
            for pos in range(first_chain_page * PS, L - TQ)
        ]
        ctx = cache_tok[np.asarray(tok, np.int64), :R].astype(np.float32)
        for i in range(TQ):
            kv = np.concatenate([ctx, kv_new[b * TQ : b * TQ + i + 1].astype(np.float32)], 0)
            q = ql[b * TQ + i].astype(np.float32)  # [H, R]
            s = q @ kv.T * scale
            p = np.exp(s - s.max(-1, keepdims=True))
            out[b * TQ + i] = (p / p.sum(-1, keepdims=True)) @ kv
    return out


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize(
    "pps,n_sel", [(64, 40), (640, 300), (2200, 500)]
)  # 640: union > 510 pages; 2200: sort fallback
def test_chain_union_matches_dense_reference(pps, n_sel, kv_dtype):
    rng = np.random.default_rng(pps)
    B = 2
    total_pages = B * pps + 2
    shape = get_kv_cache_shape(total_pages, PS, R + 128, kv_dtype)
    cache = jnp.asarray(rng.normal(0, 0.5, shape), jnp.float32).astype(kv_dtype)
    req_pi = rng.permutation(np.arange(1, B * pps + 1)).astype(np.int32)
    L = pps * PS - 5  # chain ends inside the last page
    req_seq = np.full((B,), L, np.int32)
    first_chain_page = (L - TQ) // PS
    # each chain query hits its own random set; together they cover ~all earlier pages
    hits = np.full((B * TQ, 511), -1, np.int32)
    for r in range(B * TQ):
        sel = np.sort(rng.choice(first_chain_page, n_sel, replace=False))
        hits[r, :n_sel] = sel
    ql = jnp.asarray(rng.normal(0, 0.5, (B * TQ, H, R)), jnp.bfloat16)
    kv_new = jnp.asarray(rng.normal(0, 0.5, (B * TQ, R)), jnp.float32).astype(kv_dtype)
    empty = jnp.zeros((B * TQ, H, 0), jnp.bfloat16)
    scale = 0.5  # peaky attention so a dropped page changes the output
    o, cache_out = sparse_mla_chain_request_level(
        ql,
        empty,
        kv_new,
        jnp.zeros((B * TQ, 0), kv_dtype),
        cache,
        jnp.asarray(hits),
        jnp.asarray(req_seq),
        jnp.asarray(req_pi),
        jnp.asarray([0, 0, B], jnp.int32),
        sm_scale=scale,
        page_size=PS,
        queries_per_seq=TQ,
        kv_lora_rank=R,
    )
    cache_tok = np.asarray(cache.astype(jnp.float32)).reshape(total_pages * PS, -1)
    ref = _reference(
        np.asarray(ql),
        np.asarray(kv_new.astype(jnp.float32)),
        cache_tok,
        hits,
        req_seq,
        req_pi,
        pps,
        scale,
    )
    assert np.std(ref) > 0.05  # sensitive: not an average over thousands of tokens
    np.testing.assert_allclose(np.asarray(o, np.float32), ref, rtol=2e-2, atol=2e-2)
    # the kernel itself writes the chain tokens' latent KV (no separate prewrite needed)
    out_tok = np.asarray(cache_out.astype(jnp.float32)).reshape(total_pages * PS, -1)
    for b in range(B):
        for i in range(TQ):
            pos = L - TQ + i
            slot = req_pi[b * pps + pos // PS] * PS + pos % PS
            np.testing.assert_array_equal(
                out_tok[slot, :R], np.asarray(kv_new[b * TQ + i].astype(jnp.float32))
            )


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("pps,n_sel,mask_pages", [(64, 40, 64), (640, 300, None), (640, 300, 64)])
def test_page_level_matches_dense_reference(pps, n_sel, mask_pages, kv_dtype):
    """Decode / draft path: one query per sequence over its hit pages + its own page."""
    from sgl_jax.srt.kernels.dsa.sparse_mla import sparse_mla_page_level

    rng = np.random.default_rng(pps + n_sel)
    T = 4
    total_pages = T * pps + 2
    shape = get_kv_cache_shape(total_pages, PS, R + 128, kv_dtype)
    cache = jnp.asarray(rng.normal(0, 0.5, shape), jnp.float32).astype(kv_dtype)
    page_indices = rng.permutation(np.arange(1, T * pps + 1)).astype(np.int32)
    kv_lens = np.asarray([pps * PS - 3 - 7 * t for t in range(T)], np.int32)
    hits = np.full((T, 511), -1, np.int32)
    for t in range(T):
        new_page = (kv_lens[t] - 1) // PS
        hits[t, :n_sel] = np.sort(rng.choice(new_page, n_sel, replace=False))
    ql = jnp.asarray(rng.normal(0, 0.5, (T, H, R)), jnp.bfloat16)
    kv_new = jnp.asarray(rng.normal(0, 0.5, (T, R)), jnp.float32).astype(kv_dtype)
    scale = 0.5
    cu_kv = np.arange(T + 1, dtype=np.int32) * pps * PS
    # mask_pages < pps: sequences longer than the mask take the runtime sort fallback.
    o, _ = sparse_mla_page_level(
        ql,
        jnp.zeros((T, H, 0), jnp.bfloat16),
        kv_new,
        jnp.zeros((T, 0), kv_dtype),
        cache,
        jnp.asarray(kv_lens),
        jnp.full((T, 1), -1, jnp.int32),
        jnp.asarray(page_indices),
        jnp.arange(T + 1, dtype=jnp.int32),
        jnp.asarray(cu_kv),
        jnp.asarray([T, T, T], jnp.int32),
        jnp.asarray(hits),
        sm_scale=scale,
        page_size=PS,
        pages_per_seq=pps,
        kv_lora_rank=R,
        k_pages_max=512,
        mask_pages=mask_pages,
    )
    cache_tok = np.asarray(cache.astype(jnp.float32)).reshape(total_pages * PS, -1)
    ref = np.zeros((T, H, R), np.float32)
    for t in range(T):
        L = int(kv_lens[t])
        new_page = (L - 1) // PS
        base = t * pps
        tok = [
            page_indices[base + p] * PS + j
            for p in sorted(set(hits[t, :n_sel].tolist()))
            for j in range(PS)
        ]
        tok += [page_indices[base + new_page] * PS + j for j in range(L - 1 - new_page * PS)]
        kv = np.concatenate(
            [
                cache_tok[np.asarray(tok, np.int64), :R],
                np.asarray(kv_new[t : t + 1].astype(jnp.float32)),
            ],
            0,
        )
        s = np.asarray(ql[t], np.float32) @ kv.T * scale
        p = np.exp(s - s.max(-1, keepdims=True))
        ref[t] = (p / p.sum(-1, keepdims=True)) @ kv
    assert np.std(ref) > 0.05
    np.testing.assert_allclose(np.asarray(o, np.float32), ref, rtol=2e-2, atol=2e-2)
