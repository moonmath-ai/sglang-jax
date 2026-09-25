"""Micro-benchmark: DSA indexer top-k, Pallas `streamindex_topk` vs `streamindex_topk_ref`.

Runs directly on the TPU (no model, no weights). GLM-5.3 shapes: H_I=32, D=128, k=2048,
page=128. Sweeps decode batch and context length. This is the kernel behind DSA_INDEXER_KERNEL.

From the repository root:
  PYTHONPATH=python python benchmark/kernels/dsa/bench_indexer_topk.py
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk
from sgl_jax.srt.kernels.dsa.ref import streamindex_topk_ref
from sgl_jax.srt.kernels.mla.v2.kernel import align_to

H_I, D, K, PAGE = 32, 128, 2048, 128
PAGE_SIZE_PER_KV_PACKING = PAGE // 4
KV_PACKING = 4
Q_LKV = align_to(D, 128)
RECORD = Q_LKV + (Q_LKV // 128)          # bf16 data + f32 scale per 128
WIDTH = align_to(RECORD, 128)


def make_inputs(ctx, B):
    """Decode: B sequences, 1 query each, `ctx` cached tokens each."""
    num_tokens = B
    q = jnp.zeros((num_tokens, H_I, D), jnp.float32)
    w = jnp.zeros((num_tokens, H_I), jnp.float32)
    pages_per_seq = align_to(ctx, PAGE) // PAGE
    max_blocks = pages_per_seq
    num_pages = max_blocks * B + 1        # +1 reserved page 0
    kv_cache = jnp.zeros((num_pages, PAGE_SIZE_PER_KV_PACKING, KV_PACKING, WIDTH), jnp.uint8)
    # block table: seq i owns pages 1+i*pages_per_seq .. ; page 0 reserved
    bt = np.zeros((B, max_blocks), np.int32)
    for i in range(B):
        for p in range(pages_per_seq):
            bt[i, p] = 1 + i * pages_per_seq + p
    page_indices = jnp.array(bt.flatten())
    cu_q = jnp.arange(0, B + 1, dtype=jnp.int32)
    seq_lens = jnp.full((B,), ctx, jnp.int32)
    cu_kv_lens = jnp.arange(0, B + 1, dtype=jnp.int32) * pages_per_seq
    distribution = jnp.array([B, B, B], jnp.int32)   # all decode
    return q, w, kv_cache, seq_lens, page_indices, cu_q, cu_kv_lens, distribution, pages_per_seq


def timeit(fn, iters=10):
    # warmup
    out = fn()
    jax.block_until_ready(out)
    t = time.time()
    for _ in range(iters):
        out = fn()
        jax.block_until_ready(out)
    return (time.time() - t) / iters * 1000  # ms


def main():
    print(f"TPU: {jax.devices()[0].device_kind}, H_I={H_I} D={D} k={K} page={PAGE}")
    print(f"{'ctx':>8} {'B':>3} {'ref_ms':>10} {'kern_ms':>10} {'speedup':>8}")
    for ctx in (8192, 32768, 131072):
        for B in (1, 4, 16):
            q, w, cache, seq, pi, cuq, cukv, dist, pps = make_inputs(ctx, B)
            ref_cache = jnp.zeros((cache.shape[0], PAGE, D), jnp.bfloat16)
            try:
                ref_ms = timeit(lambda: streamindex_topk_ref(
                    q, w, ref_cache, seq, pi, cuq, cukv, dist,
                    k=K, pages_per_seq=pps, one_token_per_seq=True), iters=5)
            except Exception as e:
                ref_ms = float("nan"); print("  ref err:", str(e)[:80])
            try:
                k_ms = timeit(lambda: streamindex_topk(
                    q, w, cache, seq, pi, cuq, dist,
                    k=K, compression_ratio=1, num_kv_pages_per_block=64, num_queries_per_block=1), iters=5)
            except Exception as e:
                k_ms = float("nan"); print("  kern err:", str(e)[:120])
            sp = ref_ms / k_ms if (ref_ms == ref_ms and k_ms == k_ms and k_ms > 0) else float("nan")
            print(f"{ctx:>8} {B:>3} {ref_ms:>10.3f} {k_ms:>10.3f} {sp:>8.2f}")


if __name__ == "__main__":
    main()