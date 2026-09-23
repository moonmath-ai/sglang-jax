# GLM-5.3-Flash (`glm5_next`) support for sglang-jax — design

Target: serve `zai-org/GLM-5.3-Flash` (arch `Glm5NextForConditionalGeneration`) on TPU v7 (8 chips)
through sglang-jax, reusing the existing GLM-5 / KDA / DSA machinery where possible.

This document is the implementation plan. It records exactly what exists, what is missing, what we
will add, and in what order. Verified against sglang-jax @ `eaf79799b` and transformers `5.17.0`.

## 1. Model facts (from `config.json`)

| field | value |
|---|---|
| hidden_size | 4096 |
| num_hidden_layers | 45 |
| first_k_dense_replace | 3 |
| layer_types | 34 `linear_attention` + 11 `deepseek_sparse_attention` (`[3×KDA, 1×DSA]` repeat, KDA trailing) |
| kda_layers (0-based) | 0,1,2,4,5,6,…,44 |
| full_attn_layers (0-based) | 3,7,11,15,19,23,27,31,35,39,43 |
| linear_attn_config | num_heads=64, head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0 |
| num_attention_heads | 64 |
| q_lora_rank | **1536** |
| kv_lora_rank | 512 |
| qk_nope_head_dim | **256** (GLM-5.1/5.2 use 192) |
| qk_rope_head_dim | 0 (NoPE) |
| v_head_dim | 256 |
| n_routed_experts | 288, num_experts_per_tok 8, n_shared_experts 1 |
| moe_intermediate_size | 2048, intermediate_size 12288 |
| index_n_heads 32, index_head_dim 128, index_topk 2048, index_kpool 4 |
| hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6 (mHC) |
| swiglu_limit 10.0, routed_scaling_factor 2.5, norm_topk_prob true |
| vocab_size 154880 |

## 2. What already exists in sglang-jax (reuse)

- **KDA backend + kernels** (`layers/attention/linear/kda_backend.py`, `kernels/kda/`): model-agnostic.
  Reads per-layer attributes off a `RadixLinearAttention` (`q_conv1d/k_conv1d/v_conv1d`, `A_log`,
  `dt_bias`, `num_q_heads/head_*_dim`, `scale`, `kda_gate_lower_bound`). Accepts `a`=raw gate, `b`=beta.
- **`BailingKDAAttention`** (`models/bailing_moe_v3.py`): the closest template — but Ling-3 uses
  *direct* `f_proj`/`g_proj`. GLM-5.3 uses **LoRA pairs** `f_a/f_b` (gate) and `g_a/g_b` (output gate),
  like `KimiDeltaAttention`.
- **MLA + DSA** (`glm5_moe.py`, `layers/attention/dsa_sparse_backend.py`): absorbed MLA + streamindex
  top-k indexer, `DSAFusedCache` (kv, idx, topk, topk_pages) with IndexShare threading.
- **Hybrid linear/full dispatch** (`hybrid_linear_attn_backend.py`): two sub-backends keyed by
  `full_attn_layers`; `attn_backend_wrapper` selects `KDAAttnBackend` for Kimi / Ling (`use_kda`).
- **Pools**: `RecurrentStatePool` + `HybridLinearKVPool` (wraps a `MLATokenToKVPool`) + `MemoryPools`.
- **MoE**: `GateLogit`/`TopK`/`FusedEPMoEV2`/`EPMoE`, shared experts, weight mappings via
  `create_moe_weights_mapping`.
- **mHC kernel** (`kernels/mhc/`, `layers/hyperconnection.py`): exists but no model consumes it.
- **Serving**: scheduler, paged KV, OpenAI/Anthropic API — production.

## 3. Gaps to close

1. **No `Glm5Next*` model.** `glm5_moe.py` is GLM-5.1/5.2 (MLA+DSA only; hardcoded MLA dims
   `q_lora=2048 qk_nope=192`; no KDA; no mHC; no MTP; no vision).
2. **KDA wiring for GLM.** Need a `Glm5NextKDA` module with `f_a→f_b` gate, `g_a→g_b` output gate,
   `b_proj` beta, `o_norm`, `o_proj`, direct `q/k/v` + split convs.
3. **MLA dims hardcoded** in `Glm5Attention` — parameterize from config (q_lora 1536, qk_nope 256, rope 0).
4. **Hybrid + DSA is rejected.** Two guards (`model_runner_kv_cache_mixin.py:637` assert,
   `:660-669` `_validate_kv_pool_compatibility`) and `HybridLinearKVPool` lacks
   `get_indexer_key_buffer` forwarding.
5. **`attn_backend_wrapper`** doesn't recognize a GLM-5.3 config.
6. **mHC not integrated** into any decoder layer.
7. **Registry/config detection** for `glm5_next` / `Glm5NextForConditionalGeneration`.

Out of scope for milestone 1: **MTP draft layer**, **vision tower**. Add later.

## 4. Design

### 4.1 Config (`configs/glm5_next.py`)
- Load the HF text config via transformers `Glm5NextTextConfig` (transformers 5.x).
- Detect on `architectures` containing `Glm5NextForConditionalGeneration` or `model_type == "glm5_next"`.
- Expose: `is_kda_layer(i)` from `layer_types[i] == "linear_attention"`, `linear_layer_ids`,
  `full_attention_layer_ids`, and `linear_attn_config` normalized from the HF nested dict.
- `use_kda = True` so `attn_backend_wrapper` can branch; also add an explicit branch keyed on this
  config type so detection doesn't depend on `use_kda` alone.
- Publish `indexer_types` restricted to the full-attention layers for DSA sizing.

### 4.2 KDA module (`Glm5NextKDA`, new)
Mirrors `KimiDeltaAttention` but with GLM's parameterization and `kda_lower_bound` from config:
- Projections: `q,k,v → num_heads*head_dim`; convs `q_conv1d/k_conv1d/v_conv1d` `[proj,4]`.
- Gate: `raw_gate = f_b_proj(f_a_proj(h))` (LoRA), `beta = sigmoid(b_proj(h))`.
- Output gate: `og = g_b_proj(g_a_proj(h))`, applied as `o_norm(o) * silu(og)` before `o_proj`.
- `RadixLinearAttention(..., kda_lower_bound=cfg.gate_lower_bound, A_log, dt_bias)`.
- `head_q_dim == head_k_dim == head_v_dim == 128` (equal ⇒ conv packing valid).

### 4.3 Attention dispatch (per layer)
- `linear_attention` → `Glm5NextKDA` → `recurrent_state_pool`.
- `deepseek_sparse_attention` → parameterized `Glm5Attention` (absorbed MLA + DSA indexer) →
  `token_to_kv_pool`.

### 4.4 Hybrid + DSA pool changes (`memory_pool.py`, `model_runner_kv_cache_mixin.py`)
- Remove/relax the two guards; require the inner pool to be `MLATokenToKVPool` with
  `indexer_key_dim`/`num_indexer_layers` sized to the **full-attention subset**.
- Add `HybridLinearKVPool.get_indexer_key_buffer(slot)` (and any other DSA accessor) delegating to
  `self.full_kv_pool`.
- Ensure `_dsa_indexer_cache_params()` counts indexer layers among `full_attention_layer_ids`.

### 4.5 mHC
- Add a `HyperConnection` module (fn/base/scale + Sinkhorn) around each sublayer, 4 streams, matching
  the verified math in `glm53/model.py::hc_pre/hc_post`. Verify against the kernel in
  `layers/hyperconnection.py`; prefer reusing it if the math matches, else wrap the Pallas kernel.

### 4.6 Model + registry
- `Glm5NextDecoderLayer` (KDA or MLA + MoE/dense MLP + mHC), `Glm5NextModel`,
  `Glm5NextForCausalLM`; `EntryClass` appended; `patch_model_config` sets MLA arch, head_dim 256,
  v_head_dim 256, ignored layers for indexer, and registers the arch for fused MoE.
- Weight mappings: `checkpoint.py` naming from the kaggle engine is the source of truth for HF names.

### 4.7 Correctness strategy
- Build tiny random weights matching the HF layout; compare `Glm5NextTextModel` (transformers) forward
  vs the sglang-jax model on CPU/single-device. The kaggle engine already passes this test
  (`test_tiny_vs_hf.py`), so it is a trusted oracle.

## 5. Order of work

1. Config + registry detection (smallest, unblocks the rest).
2. Parameterize MLA dims.
3. KDA module.
4. Hybrid + DSA pool unblock.
5. `attn_backend_wrapper` branch.
6. mHC integration.
7. Model class + weight mappings.
8. Tiny correctness test → then the real checkpoint.

## 6. Risk register

| risk | severity | mitigation |
|---|---|---|
| transformers 5.x vs sglang-jax pin 4.57 | med | config/tokenizer APIs are stable; verify imports; pin 5.x only where glm5_next is needed |
| KDA kernel assumes `v_head == k_head` & conv `D % 3 == 0` | low | GLM-5.3 has equal head dims 128 |
| DSA indexer cache sizing under hybrid | high | restrict `num_indexer_layers` to full-attn subset; unit-test pool shapes |
| mHC math mismatch vs kernel | med | test against `glm53` verified implementation |
| NoPE (rope_head_dim 0) in MLA/indexer paths | med | GLM-5.1 uses rope; indexer `rope_dim=64` hardcoded — verify GLM-5.3 indexer RoPE usage |
| Memory: 288 experts × 45 layers | med | MoE is EP-sharded; validate on v7 |