
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

## More experiments (ep-size / backend)
- `--ep-size 8` (epmoe): **2.0 tok/s decode** — catastrophic. EP all-to-all dispatch over 8
  chips with only 36 experts/chip is far worse than replication.
- `--moe-backend fused_v2 --ep-size 8`: 1.6 tok/s — same problem.
- => **ep_size=1 (experts replicated, psum(expert) combine) is the best MoE config here.**
- GMM micro-bench: 3 GMMs/call ~0.05 ms each (tile choice only ~±10%). So the matmuls are
  cheap; the EPMoE overhead (~1.6 ms/call in the standalone bench) is permute (argsort+bincount
  over tokens*8) + gather + psum(expert) combine, NOT the GMMs. Standalone bench includes host
  dispatch, so treat the absolute as an upper bound.

## Open question
Fixed 18.7ms/step: is it MoE overhead, MLA attention, mHC, or host/launch? Kernel micro-benches
overstate per-layer cost (host dispatch). Need an in-model per-layer timing or a proper device
trace with named scopes to attribute reliably.

## Proper device-kernel attribution (jax.profiler host_tracer_level=1, decode-heavy)
Aggregate device_duration_ps by kernel family (40 decode tok + prefill window):
| family | total ms | count |
|---|---|---|
| gmm_v2 (MoE experts) | 207 | 6048 |
| fusion (elementwise: norms, silu, routing, mHC) | 196 | 201428 |
| gather_fusion (expert gather) | 97 | 8286 |
| all-reduce | 71 | 3064 |
| all-gather | 62 | 4096 |
| constant_dynamic-slice_fusion | 60 | 3984 |
| quantized_matmul_kernel (dense proj) | 34 | 8674 |
| top_k (DSA indexer) | 30 | 2203 |
| collective-permute | 25 | 15256 |
| psum | 25 | 2264 |

**No single kernel dominates.** Fixed step cost is distributed across 45 layers:
- MoE compute (gmm 207) + expert gather (97) = ~304
- **Collectives (all-reduce 71 + all-gather 62 + permute 25 + psum 25) = ~183**
- Elementwise fusion 196 (norms/silu/mHC/routing)
- Dense attention matmuls only 34; indexer top_k only 30.

### Implication
The ~18.7ms/step is not one slow kernel — it's the per-layer overhead of a 45-layer MoE
(export dispatch collectives + gather + elementwise) at tp8/ep1. Levers: reduce collective
count/frequency, fuse elementwise, or reduce gather cost. ep8 (all-to-all) was worse.
