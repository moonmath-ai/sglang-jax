# GLM-5.3-Flash TPU v7 baseline (real FP8 weights, KPool off)

Config: tp=8, dsa_sparse, page 128, disable-radix-cache, bf16 activations, fp8 expert/attn weights.
KV: 76,800 tokens, 1.21 GB (latent 1.01 + DSA indexer 0.20). compile cache warm.

| Phase | tok/s |
|---|---|
| prefill ~1.1k | 4,422 |
| prefill ~4.5k | 9,261 |
| prefill ~8.9k | 9,838 |
| decode c=1 | 57.6 |

Commit: e05aeef (branch glm53-flash-tpu)

## Concurrency (512 in / 512 out)
| c | agg tok/s | per-stream | TTFT ms |
|---|---|---|---|
| 1 | 57.4 | 57.4 | 209 |
| 2 | 82.8 | 41.4 | 318 |
| 4 | 85.3 | 21.3 | 408 |
| 8 | 111.8 | 14.0 | 520 |
| 16 | 144.6 | 9.0 | 742 |

## DSA_INDEXER_KERNEL=1 (Pallas indexer top-k vs jnp ref) — dummy weights, 512/512
| c | OFF agg | ON agg | speedup |
|---|---|---|---|
| 1 | 57.4 | 97.6 | 1.7x |
| 2 | 82.8 | 186.6 | 2.3x |
| 4 | 85.3 | 1.7x |
| 4 | 85.3 | 337.5 | 4.0x |
| 8 | 111.8 | 555.5 | 5.0x |
| 16 | 144.6 | 913.1 | 6.3x |

Context sweep decode (indexer ON, dummy): 16tok 87.5, 512 87.7, 2k 70.1, 8k 52.0, 16k 35.1.
Still context-degrading => a second context-proportional cost remains (attention/topk, not indexer).
