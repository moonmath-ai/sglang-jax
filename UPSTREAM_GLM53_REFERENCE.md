# Upstream sglang GLM-5.3-Flash reference (for the sglang-jax port)

Cloned `sgl-project/sglang` to `/home/emir/sglang`. Key files:

| File | What it is |
|---|---|
| `python/sglang/srt/models/glm5_next.py` | The model: `Glm5NextLinearAttention` (KDA), `Glm5NextDecoderLayer` (mHC + MoE), `Glm5NextForConditionalGeneration`, vision, weight loading |
| `python/sglang/srt/models/glm5_next_nextn.py` | MTP (NextN) draft layer |
| `python/sglang/srt/configs/glm5_next.py` | `Glm5NextTextConfig` + `is_kda_layer` / `linear_layer_ids` / `full_attention_layer_ids` |
| `python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py` | `IndexerKPool` — the KPool indexer |
| `python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py` | `kpool_softmax_rotate_write_cache` — the actual pooling + fp8 compressed-cache write |
| `python/sglang/srt/layers/attention/dsa/dsa_backend_kpool.py` | KPool backend (metadata, paged schedules) |

## Model structure (authoritative)

- **Layer split**: `config.is_kda_layer(i)` = `i in linear_attn_config["kda_layers"]`. KDA layers use
  `Glm5NextLinearAttention`; the rest use `DeepseekV2AttentionMLA(..., skip_rope=True)` (NoPE!).
- **KDA** (`Glm5NextLinearAttention`): `num_k_heads == num_v_heads == num_heads == 64`,
  `head_dim = 128`, conv kernel 4. Projections `q/k/v_proj`, `q_conv1d/k_conv1d/v_conv1d`,
  `b_proj` (beta), `f_a_proj→f_b_proj` (gate LoRA), `g_a_proj→g_b_proj` (output gate).
  Upstream fuses some of these (`fused_qkvbfg_a_proj`, `fused_fg_b_proj`) only when unquantized;
  the released checkpoint ships the **per-tensor** names, so the unfused path applies.
- **CHECKPOINT PREFIX**: tensors are `model.language_model.layers.N.*`; upstream simply does
  `name = name.replace("language_model.", "")` so it becomes `model.layers.N.*`.
- **mHC** (`Glm5NextDecoderLayer._hc_pre` / `hc_post` / `hc_ffn_post_pre`): four-stream
  manifold-constrained hyper-connections.
- **NoPE**: DSA MLA runs with `skip_rope=True` (qk_rope_head_dim == 0).

## KPool (indexer compression) — exact math

`kpool_softmax_rotate_write_cache(pool, buf, slot_k, slot_score, ape, loc, ...)`:
- `slot_k`: `[pool_size, head_dim]` bf16 (the `index_kpool` consecutive keys of one group)
- `slot_score`: **same shape as slot_k** `[pool_size, head_dim]` — the compress gate, per (pos, dim)
- `ape`: `[pool_size, head_dim]` fp32 position embedding
- Writes a **compressed, fp8** pooled key per group to a paged cache (`BLOCK_SIZE_K` page).
- Pooled key = `softmax_over_pool(slot_score + ape) · slot_k`, then rotated + fp8-quantized.

This matches the model-side `GlmDsaIndexer.pool_keys` already implemented in sglang-jax
(`gate = index_kpool_compress_gate(x)` shape `[T, head_dim]`, `ape` `[kpool, head_dim]`,
`softmax(gate+ape, axis=pool)` weighted sum).

## What upstream has that sglang-jax still needs (KPool, for TPU)

1. **Compressed indexer cache**: pooled keys stored per `index_kpool` tokens (fp8), with the
   paged/MQA-logits scatter accounting for the compression ratio.
2. **Compressed seq_len / causal bound**: a query can only see pooled slot `e` once
   `e < (q_pos+1)//index_kpool` (the DeepSeek-V4 indexer rule) — the sglang-jax
   `streamindex_topk` non-page path already implements this via `compression_ratio`.
3. **Tail tokens**: `index_kpool_always_select_tail` appends the incomplete trailing slot's raw
   tokens to the top-k (`get_dsa_mtp_topk_width` = `index_topk + index_kpool - 1`).
4. fp8 pooling + Hadamard rotation before the compressed cache write.

The **model side** is done in sglang-jax; the **cache/scatter/top-k** is the remaining TPU work.
The sglang-jax `streamindex_topk` Pallas kernel already supports `compression_ratio > 1` in its
non-page path, which is the intended extension point.