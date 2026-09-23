"""Tune GMM v2 tile sizes (tile_m, tile_k, tile_n) for GLM-5.3-Flash on TPU v7.

GLM-5.3 EPMoE shapes (tp=8, ep=1, fp8 weights):
  GEMM1 wi: [m, 4096] @ [288, 4096, 256]     (k=4096, n=256)
  GEMM2 wo: [m, 256]  @ [288, 256, 4096]     (k=256,  n=4096)
for m in {32 (decode bs=1..4), 128, 512} (m = tokens*topk padded).

Writes the best tiles per shape into TUNED_TILE_SIZES_GMM_V2["TPU v7"].

Usage: python -m benchmark.kernels.megablox_gmm.tune_gmm_v2_glm53
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
    TileSizes,
    is_supported_by_gmm_v2,
)
from sgl_jax.srt.utils.jax_utils import get_device_name

G = 288
FP8 = jnp.float8_e4m3fn
BLOCK = 128  # quantization block along k
LANES = 128  # tile align

# (size_m, size_k, size_n, label)
SHAPES = [
    (32, 4096, 256, "decode bs1-4 GEMM1"),
    (32, 256, 4096, "decode bs1-4 GEMM2"),
    (128, 4096, 256, "decode bs16 GEMM1"),
    (128, 256, 4096, "decode bs16 GEMM2"),
    (512, 4096, 256, "prefill GEMM1"),
    (512, 256, 4096, "prefill GEMM2"),
]


def make_inputs(m, k, n):
    lhs = jnp.zeros((m, k), jnp.bfloat16)
    rhs = jnp.zeros((G, k, n), FP8)
    nblocks = (k + BLOCK - 1) // BLOCK
    scale = jnp.ones((G, nblocks, 1, n), jnp.float32)
    gs = jnp.full((G,), m // G, jnp.int32)
    rem = m - int(gs[0]) * G
    if rem:
        gs = gs.at[-1].add(rem)
    return lhs, rhs, scale, gs


def _candidates(m):
    tms = sorted({x for x in (32, 64, 128, 256) if x <= max(m, 32) and x % 32 == 0})
    tks = [k for k in (64, 128, 256, 512, 1024) ]
    tns = [n for n in (32, 64, 128, 256, 512)]
    for tm, tk, tn in itertools.product(tms, tks, tns):
        yield tm, tk, tn


def bench(m, k, n):
    lhs, rhs, scale, gs = make_inputs(m, k, n)
    goff = jnp.array(0, jnp.int32)
    best = (None, float("inf"))
    results = []
    for tm, tk, tn in _candidates(m):
        if tk % BLOCK != 0:
            continue
        if n % tn != 0 and tn % n != 0:
            pass
        try:
            fn = lambda: gmm(lhs, rhs, gs, preferred_element_type=jnp.bfloat16,
                             rhs_scale=scale, group_offset=goff,
                             maybe_quantize_lhs=True, acc_dtype=jnp.float32,
                             v2_tile_info=TileSizes(tile_m=tm, tile_k=tk, tile_n=tn))
            ms_list = multiple_iteration_timeit_from_trace(fn, lambda: (), f"gmm_v2-g_{G}-m_{m}-k_{k}-n_{n}", tries=4,
                                                           trace_root="/tmp/opencode/gmmtrace")
            ms = float(np.mean(ms_list)) if ms_list else float("nan")
        except Exception as e:
            continue
        if ms == ms:
            results.append((ms, tm, tk, tn))
            if ms < best[1]:
                best = ((tm, tk, tn), ms)
    results.sort()
    return best, results[:3]


def main():
    print("device:", get_device_name(), "G:", G)
    out = {}
    for m, k, n, label in SHAPES:
        best, top = bench(m, k, n)
        print(f"\n{label}: g={G} m={m} k={k} n={n}")
        for ms, tm, tk, tn in top:
            print(f"   {ms*1000:8.0f} us  tile=({tm},{tk},{tn})")
        if best[0]:
            out[(m, k, n)] = (best[0], best[1])
    print("\n=== TUNED_TILE_SIZES_GMM_V2 entries for 'TPU v7' ===")
    for (m, k, n), (tiles, ms) in out.items():
        print(f'        ("float8_e4m3fn", "float8_e4m3fn", {G}, {m}, {k}, {n}): {tiles},  # {ms*1000:.0f} us')


if __name__ == "__main__":
    main()