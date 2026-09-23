# GLM-5.3-Flash TPU v7 decode cost decomposition (real FP8 weights)

## Method
End-to-end decode tok/s vs context length (trustworthy; standalone kernel benches overstate
per-layer cost because they include host dispatch not present in the fused jitted loop).
Fit: `step_ms = 18.7 + 1.345 * (ctx/1000)`.

| ctx | step ms |
|---|---|
| 17 | 18.3 |
| 513 | 19.1 |
| 2049 | 22.7 |
| 8193 | 29.0 |
| 16385 | 41.0 |

## Decomposition
- **Fixed per-step: 18.7 ms** (53 tok/s ceiling) = 45-layer forward (matmul + MoE + mHC + launches).
  **DOMINANT at short/medium context.** Not addressed by KV/caching work.
- **Context-proportional: 1.345 ms per 1k tokens** = MLA sparse attention over the growing cache.
  Becomes dominant past ~14k context. This is what KPool / FP8-KV / sparse-attn tuning target.

## Kernel micro-benches (context, NOT enough alone — overstate per-layer cost)
- DSA indexer top-k: Pallas `streamindex_topk` 0.20-0.89 ms (2-3x faster than the jnp ref at
  high batch/ctx) — but only ~2-10% of the step, so not the main lever.
- EPMoE standalone: ~1.0 ms (1 tok) / ~1.6 ms (16 tok) incl. dispatch; real per-layer cost
  inside the jitted loop is ~0.4 ms (18.7ms/45 layers).

## Corrected conclusion
The earlier dummy-weight A/B (showing 6x from DSA_INDEXER_KERNEL) was misleading: with zero
weights the matmul/MoE cost vanished so the indexer's relative share exploded. With real
weights the indexer is minor. **The dominant decode cost is the fixed 18.7 ms/step forward.**
