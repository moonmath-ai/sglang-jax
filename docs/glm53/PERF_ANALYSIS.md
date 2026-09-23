
# GLM-5.3-Flash TPU v7 decode bottleneck analysis (measured)

## Reliable decode decomposition (end-to-end, real weights)
`step_ms = 18.7 (fixed) + 1.345 * ctx/1000`
- Fixed 18.7 ms/step (53 tok/s ceiling): 45-layer forward.
- Context term 1.35 ms/1k: MLA sparse attention.

## Device trace (jax.profiler, host_tracer_level=1) — per-op device time, real decode
Top device-time families (whole trace window, moe_backend=epmoe, tp=8):
| family | total ms | count | avg us |
|---|---|---|---|
| gmm_v2 g=288 (MoE grouped matmul) | ~159 | 336* | ~200k |
| all-reduce | 64.9 | 2250 | 28.8 |
| gather_fusion | 59.1 | 7218 | 8.2 |
| scatter_offload / collective-permute | ~35 | | |
| top_k (indexer) | 14.0 | 1100 | 12.7 |

=> **MoE (grouped matmul + expert gather) + cross-device collectives (all-reduce) dominate the
fixed cost.** The indexer top-k is minor (14 ms over the window).

## Experiments tried
- `SGLANG_JAX_AOT_DISPATCH=auto` (removes O(n_args) per-step host dispatch): decode 55.2 vs 57.4
  baseline — **no gain**. Host dispatch is NOT the GLM-5.3/TP8 bottleneck (upstream saw it on
  tp64/753B; our arg count 1644 + tp8 is proportionally cheap).
- `SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD=1` on top: still ~55 — no gain.
- `--moe-backend fused_v2` (ep1): 9.6 tok/s — the fused kernel treats the 2D mesh as EP; with
  ep_size=1 it's wrong.
- `--moe-backend fused_v2 --ep-size 8`: **1.6 tok/s** — catastrophic; fused block config
  untuned for 288 experts / hidden 4096. Not viable without tuning.
- **epmoe (default) remains best: 55-57 tok/s.**

## Next targets (ranked)
1. EPMoE grouped matmul (`gmm_v2 g=288`) block tuning for hidden 4096 / moe_inter 2048 on v7.
2. Reduce MoE all-reduce / dispatch traffic (expert sharding layout, EP=1 means experts
   replicated across the 8 tensor shards + psum(expert) reduce).
3. MLA tuned-block-size LOOKUP MISS for our decode shape — use a tuned config (currently the
   hardcoded default).
