"""GLM-5.3-Flash (`glm5_next`) model for sglang-jax.

Architecture: 45 layers of `[3x KDA linear attention, 1x NoPE sparse MLA]`, a 288-expert MoE
(top-8 + 1 shared), and four-stream manifold-constrained hyper-connections (mHC).

This file composes three existing sglang-jax facilities rather than reimplementing them:

  * KDA layers  -> `KimiDeltaAttention` (models/kimi_linear.py), whose parameterization matches
                   GLM-5.3 exactly: `f_a/f_b` gate LoRA, `g_a/g_b` output gate, `b_proj` beta,
                   gated output RMSNorm, and `A_log`/`dt_bias`/`gate_lower_bound`.
  * DSA layers  -> `Glm5Attention` (models/glm5_moe.py), now parameterized for GLM-5.3's MLA dims
                   (`q_lora 1536`, `qk_nope 256`, `qk_rope 0`, `v 256`) and the K-pool indexer.
  * MoE         -> the same `GateLogit`/`TopK`/`EPMoE`/`FusedEPMoE(V2)` stack GLM-5 uses.

Per-layer routing (KDA -> `recurrent_state_pool`, DSA -> `token_to_kv_pool`) mirrors
`KimiDecoderLayer`; the DSA `DSAFusedCache` threading mirrors `Glm5Model`.

mHC (`hc_mult=4`) is implemented by `Glm5NextHC`, wrapping each sublayer; see the module docstring
there for the math and its provenance.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig, get_glm5_next_config
from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import FusedEPMoEV2
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.glm5_moe import (
    Glm5Attention,
    Glm5MLP,
    _requantize_glm5_shared_expert,
    build_moe_sublayer,
)
from sgl_jax.srt.models.kimi_linear import KimiDeltaAttention
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Glm5NextHC(nnx.Module):
    """Four-stream manifold-constrained hyper-connection (mHC) around one sublayer.

    Math (verified against `glm53/model.py::hc_pre`/`hc_post`):

        flat   = rmsnorm_f32(streams.reshape(..., hc*D))            # no learned weight
        mix    = flat @ fn                                          # [..., (2+hc)*hc]
        pre_w, post_w, comb_w = split(mix)
        pre    = sigmoid(pre_w * scale[0] + base[:hc]) + hc_eps
        post   = 2 * sigmoid(post_w * scale[1] + base[hc:2hc])
        comb   = softmax(comb_w.reshape(hc,hc) * scale[2] + base[2hc:]) + hc_eps
        comb   = sinkhorn(comb, iters)
        h      = sum(pre * streams)                                 # collapse to [..., D]
        ... sublayer(h) -> y ...
        streams = post[..., None] * y[:, None, :] + comb @ streams  # write back

    `fn` is stored [hc*D, (2+hc)*hc] (JAX in/out); `base` [(2+hc)*hc]; `scale` [3]. A per-Sinkhorn
    normalisation is done with explicit slice adds (as in the reference) so the chain stays one
    fusion.
    """

    def __init__(self, hidden_size: int, hc_mult: int, hc_iters: int, hc_eps: float,
                 mesh, dtype=jnp.bfloat16):
        self.hc = hc_mult
        self.hc_iters = hc_iters
        self.hc_eps = hc_eps
        self.eps = 1e-5
        n = (2 + hc_mult) * hc_mult
        sh = NamedSharding(mesh, P(None, None)) if mesh is not None else None
        sh1 = NamedSharding(mesh, P(None)) if mesh is not None else None
        self.fn = nnx.Param(jnp.zeros((hc_mult * hidden_size, n), dtype=dtype, out_sharding=sh))
        self.base = nnx.Param(jnp.zeros((n,), dtype=jnp.float32, out_sharding=sh1))
        self.scale = nnx.Param(jnp.zeros((3,), dtype=jnp.float32, out_sharding=sh1))

    def _mix(self, streams: jax.Array):
        """streams: [..., hc*D] (hc streams concatenated on the last axis)."""
        hc = self.hc
        lead = streams.shape[:-1]
        d = streams.shape[-1] // hc
        flat = streams.reshape(*lead, hc * d).astype(jnp.float32)
        flat = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + self.eps)
        mix = flat @ self.fn.value.astype(jnp.float32)
        pre_w, post_w, comb_w = mix[..., :hc], mix[..., hc:2 * hc], mix[..., 2 * hc:]
        base, scale = self.base.value.astype(jnp.float32), self.scale.value.astype(jnp.float32)
        pre = jax.nn.sigmoid(pre_w * scale[0] + base[:hc]) + self.hc_eps
        post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + base[hc:2 * hc])
        comb = jax.nn.softmax(
            comb_w.reshape(*lead, hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc), -1
        ) + self.hc_eps
        rsum = lambda c: sum(c[..., :, j:j + 1] for j in range(hc))
        csum = lambda c: sum(c[..., j:j + 1, :] for j in range(hc))
        comb = comb / (csum(comb) + self.hc_eps)
        for _ in range(self.hc_iters - 1):
            comb = comb / (rsum(comb) + self.hc_eps)
            comb = comb / (csum(comb) + self.hc_eps)
        streams_htd = streams.reshape(*lead, hc, d).astype(jnp.float32)
        collapsed = jnp.einsum("...h,...hd->...d", pre, streams_htd).astype(streams.dtype)
        return post, comb, collapsed

    def pre(self, streams: jax.Array):
        post, comb, collapsed = self._mix(streams)
        return collapsed, (post, comb)

    def post(self, y: jax.Array, state, streams: jax.Array):
        """y: [..., D] sublayer output. Returns the new [..., hc*D] streams."""
        post, comb = state
        hc = self.hc
        lead = streams.shape[:-1]
        d = streams.shape[-1] // hc
        streams_htd = streams.reshape(*lead, hc, d).astype(jnp.float32)
        y = y.astype(jnp.float32)
        out = post[..., None] * y[..., None, :] + jnp.einsum("...ij,...id->...jd", comb, streams_htd)
        return out.reshape(*lead, hc * d).astype(streams.dtype)


class Glm5NextDecoderLayer(nnx.Module):
    def __init__(self, config: Glm5NextConfig, mesh, layer_id: int = 0, dtype=jnp.bfloat16):
        self.config = config
        self.layer_id = layer_id
        self.is_kda = config.is_kda_layer(layer_id)
        self.hc_mult = config.hc_mult
        self.use_mhc = config.hc_mult and config.hc_mult > 1
        self.hidden_size = config.hidden_size

        if self.use_mhc:
            self.hc_attn = Glm5NextHC(config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters,
                                      config.hc_eps, mesh, dtype)
            self.hc_ffn = Glm5NextHC(config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters,
                                     config.hc_eps, mesh, dtype)

        # attention
        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                config=config, layer_idx=layer_id, mesh=mesh, dtype=dtype,
                v_head_dim=config.linear_attn_config["head_dim"],
            )
        else:
            indexer_types = config.indexer_types
            indexer_type = "full" if indexer_types is None else indexer_types[layer_id]
            self.self_attn = Glm5Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                mesh=mesh,
                rope_theta=getattr(config, "rope_theta", 800000.0),
                rope_scaling=getattr(config, "rope_scaling", None),
                rms_norm_eps=config.rms_norm_eps,
                use_qk_norm=config.use_qk_norm,
                layer_id=layer_id,
                dtype=dtype,
                use_absorbed=True,
                has_indexer=(indexer_type == "full"),
                indexer_type=indexer_type,
                use_dsa_sparse=True,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                index_head_dim=config.index_head_dim,
                index_n_heads=config.index_n_heads,
                indexer_rope_dim=getattr(config, "indexer_rope_dim", 0),
                index_kpool=getattr(config, "index_kpool", 1),
                index_kpool_compress=getattr(config, "index_kpool_compress", False),
            )

        # norms
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps,
                                       param_dtype=dtype, scope_name="input_layernorm")
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps,
                                                param_dtype=dtype, scope_name="post_attention_layernorm")

        # MLP
        if layer_id < config.first_k_dense_replace:
            self.mlp = Glm5MLP(config.hidden_size, config.intermediate_size, mesh, layer_id, dtype,
                               use_fused=getattr(config, "_sgl_use_fused_mlp", False))
            self.is_moe_layer = False
            self.moe_gate = None
            self.topk = None
            self.shared_experts = None
        else:
            build_moe_sublayer(self, config, mesh, layer_id, dtype, use_fused_mlp_default=False)

    def __call__(self, positions, hidden_states, forward_batch, token_to_kv_pool, recurrent_state_pool,
                 residual=None, dispatch_info=None, dsa_topk_in=None, dsa_topk_pages_in=None):
        use_mhc = self.use_mhc

        # ---- attention sublayer ----
        if use_mhc:
            h, hc_state = self.hc_attn.pre(hidden_states)
            h = self.input_layernorm(h)
        else:
            if residual is None:
                residual = hidden_states
                h = self.input_layernorm(hidden_states)
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states
                h = self.input_layernorm(hidden_states)

        if self.is_kda:
            attn_out, attn_state = self.self_attn(positions, h, forward_batch, recurrent_state_pool)
        else:
            attn_out, attn_state = self.self_attn(
                positions, h, forward_batch, token_to_kv_pool,
                dsa_topk_in=dsa_topk_in, dsa_topk_pages_in=dsa_topk_pages_in,
            )

        if use_mhc:
            hidden_states = self.hc_attn.post(attn_out, hc_state, hidden_states)
            h, hc_state = self.hc_ffn.pre(hidden_states)
            h = self.post_attention_layernorm(h)
        else:
            hidden_states = attn_out + hidden_states
            residual = hidden_states
            h = self.post_attention_layernorm(hidden_states)

        # ---- MLP sublayer ----
        if self.is_moe_layer:
            shared_output = self.shared_experts(h) if self.shared_experts is not None else None
            router_logits = self.moe_gate(h)
            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(router_logits, correction_bias, dispatch_info=dispatch_info)
            mlp_kwargs = {}
            # GLM-5.3-Flash MV2 uses a SwiGLU activation clamp on both the routed
            # and (in-kernel) shared experts. Only the V2 fused kernel supports it.
            swiglu_limit = getattr(self.config, "swiglu_limit", None)
            if swiglu_limit is not None and self.moe_backend == MoEBackend.FUSED_V2:
                mlp_kwargs["swiglu_limit"] = swiglu_limit
                mlp_kwargs["shared_swiglu_limit"] = swiglu_limit
            mlp_out = self.mlp(h, topk_weights, topk_ids, **mlp_kwargs)
            if shared_output is not None:
                mlp_out = mlp_out + shared_output
        else:
            mlp_out = self.mlp(h)
            topk_ids = None

        if use_mhc:
            hidden_states = self.hc_ffn.post(mlp_out, hc_state, hidden_states)
        else:
            hidden_states = mlp_out + hidden_states
            residual = hidden_states

        return hidden_states, residual, attn_state, topk_ids


class Glm5NextModel(nnx.Module):
    def __init__(self, config: Glm5NextConfig, mesh, dtype=jnp.bfloat16):
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embed(num_embeddings=config.vocab_size, features=config.hidden_size,
                                  dtype=dtype, param_dtype=dtype, kernel_axes=("tensor", None), mesh=mesh)
        self.layers = nnx.data([
            Glm5NextDecoderLayer(config=config, layer_id=i, dtype=dtype, mesh=mesh)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps,
                            param_dtype=dtype, scope_name="norm")

    def __call__(self, forward_batch: ForwardBatch, memory_pools):
        from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSAFusedCache

        hidden_states = self.embed_tokens(forward_batch.input_ids)
        use_mhc = self.layers[0].use_mhc
        if use_mhc:
            # mHC keeps hc parallel residual streams: broadcast the embedding across the hc axis.
            hc = self.layers[0].hc_mult
            hidden_states = jnp.tile(hidden_states, (1, hc))
        residual = None
        layers_kv_fused, layers_idx_fused, layers_topk_ids = [], [], []
        layers_recurrent, layers_conv = [], []
        dsa_topk, dsa_topk_pages = None, None

        for layer in self.layers:
            hidden_states, residual, attn_state, topk_ids = layer(
                forward_batch.positions, hidden_states, forward_batch,
                memory_pools.token_to_kv_pool, memory_pools.recurrent_state_pool,
                residual, dispatch_info=forward_batch.expert_location_metadata,
                dsa_topk_in=dsa_topk, dsa_topk_pages_in=dsa_topk_pages,
            )
            if layer.is_kda:
                rec_buf, conv_buf_list = attn_state
                layers_recurrent.append(rec_buf)
                layers_conv.append(conv_buf_list)
            elif isinstance(attn_state, DSAFusedCache):
                layers_kv_fused.append(attn_state.kv)
                if attn_state.idx is not None:
                    layers_idx_fused.append(attn_state.idx)
                if attn_state.topk is not None:
                    dsa_topk = attn_state.topk
                if attn_state.topk_pages is not None:
                    dsa_topk_pages = attn_state.topk_pages
            else:
                layers_kv_fused.append(attn_state)
            layers_topk_ids.append(topk_ids)

        if use_mhc:
            # collapse the hc streams into one before the final norm: mean over the hc axis
            hc = self.layers[0].hc_mult
            d = hidden_states.shape[-1] // hc
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hc, d).mean(axis=1)
        elif residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return (hidden_states, layers_kv_fused, layers_idx_fused,
                (layers_recurrent, layers_conv), layers_topk_ids)


class Glm5NextForConditionalGeneration(nnx.Module):
    """Serving class for `zai-org/GLM-5.3-Flash` (HF arch `Glm5NextForConditionalGeneration`)."""
    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        from sgl_jax.srt.configs.model_config import AttentionArch

        text = getattr(config, "hf_text_config", config.hf_config)
        config.attention_arch = AttentionArch.MLA
        config.head_dim = getattr(text, "qk_nope_head_dim", 256) + getattr(text, "qk_rope_head_dim", 0)
        if hasattr(config.hf_config, "head_dim"):
            config.hf_config.head_dim = config.head_dim
        config.hf_config._sgl_use_fused_mlp = False  # fused_mlp kernel uses a removed shard_map kwarg on jax 0.11

        # Make the quant config reachable from the model's own config: the MoE layers read
        # `config.quantization_config` to decide whether to build static-quant placeholder scales.
        if getattr(config, "quantization_config", None) is not None:
            for c in (getattr(config, "hf_config", None), getattr(config, "hf_text_config", None)):
                if c is not None:
                    try:
                        c.quantization_config = config.quantization_config
                    except Exception:
                        pass

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        # Accept either the normalized Glm5NextConfig or the raw HF (multimodal) config.
        if not isinstance(config, Glm5NextConfig):
            normalized = get_glm5_next_config(config)
            if normalized is None:
                raise ValueError(f"not a glm5_next config: {type(config).__name__}")
            # carry over the runtime fields the runner/DSA path reads off hf_text_config
            cfg = normalized
            # keep indexer_types / layer_types from the HF text config if present
            text = getattr(config, "text_config", config)
            if getattr(cfg, "indexer_types", None) is None and getattr(text, "indexer_types", None) is not None:
                cfg.indexer_types = list(text.indexer_types)
            # carry the quantization config (the MoE layers need it to build static-quant scales)
            qc = getattr(config, "quantization_config", None) or getattr(text, "quantization_config", None)
            if qc is not None:
                cfg.quantization_config = qc
            # The runner sets these on the outer HF config. Normalization reads
            # the nested text config, so preserve the serving MoE choices.
            cfg.moe_backend = getattr(
                config, "moe_backend", getattr(text, "moe_backend", MoEBackend.EPMOE)
            )
            cfg.ep_size = getattr(config, "ep_size", getattr(text, "ep_size", 1))
        else:
            cfg = config
        self.config = cfg
        self.mesh = mesh
        self.dtype = dtype
        self.model = Glm5NextModel(cfg, dtype=dtype, mesh=mesh)
        if not getattr(cfg, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size, dtype=dtype,
                                          param_dtype=dtype, kernel_axes=("tensor", None))
        self.logits_processor = LogitsProcessor(cfg.vocab_size, mesh=mesh)

    def __call__(self, forward_batch: ForwardBatch, memory_pools, logits_metadata: LogitsMetadata):
        hidden, kv_fused, idx_fused, recurrent, topk_ids = self.model(forward_batch, memory_pools)
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden, self.model.embed_tokens, logits_metadata)
        kv_update = (kv_fused, idx_fused) if idx_fused else kv_fused
        return (output, {"token_to_kv_pool": kv_update, "recurrent_state_pool": recurrent}, True, topk_ids)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype)
        loader.load_weights_from_safetensors(self._create_weight_mappings(model_config))
        for layer in self.model.layers:
            if not layer.is_kda:
                layer.self_attn.post_load_weights()
            if isinstance(getattr(layer, "mlp", None), FusedEPMoEV2):
                _requantize_glm5_shared_expert(layer.mlp)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "post_load_weights"):
                layer.mlp.post_load_weights()
            if getattr(layer, "shared_experts", None) is not None and hasattr(layer.shared_experts, "post_load_weights"):
                layer.shared_experts.post_load_weights()
        logger.info("GLM-5.3-Flash weights loaded.")

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        """HF checkpoint names -> sglang-jax params.

        Reuses GLM-5.2's DSA-layer and MoE mappings (`Glm5ForCausalLM._create_moe_layer_mappings`) for the
        `deepseek_sparse_attention` layers, and adds GLM-5.3's KDA projection mappings and mHC tensors.

        Verified against `glm53/checkpoint.py` for the exact HF names.
        """
        cfg = self.config
        # The GLM-5.3-Flash checkpoint stores text tensors under `model.language_model.` (multimodal
        # wrapper); the model's own params are at `model.`. HF source keys use the former.
        HF = "model.language_model"
        mappings: dict = {
            f"{HF}.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding", sharding=("tensor", None), transpose=False
            ),
            f"{HF}.norm.weight": WeightMapping(target_path="model.norm.scale", sharding=(None,), transpose=False),
        }
        if not getattr(cfg, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and getattr(quant_config, "is_static_checkpoint", False)
        indexer_types = getattr(cfg, "indexer_types", None)

        for i in range(cfg.num_hidden_layers):
            is_dense = i < cfg.first_k_dense_replace
            has_indexer = indexer_types is None or indexer_types[i] == "full"
            # Base mappings (norms + attention + MLP/MoE) via GLM-5.2's builder, which only reads
            # self.config. For KDA layers we then swap the MLA attention keys for KDA keys.
            from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM

            layer_map = Glm5ForCausalLM._create_moe_layer_mappings(
                self, i, i, is_dense, is_static_quant, has_indexer,
                hf_prefix=f"{HF}.layers",
            )
            if cfg.is_kda_layer(i):
                layer_map = {
                    k: v for k, v in layer_map.items()
                    if ".self_attn." not in k
                }
                layer_map.update(self._create_kda_attention_mappings(i, cfg, HF))
            elif has_indexer and getattr(cfg, "index_kpool_compress", False) and cfg.index_kpool > 1:
                # GLM-5.3-Flash KPool indexer extra tensors (absent on GLM-5.1/5.2).
                hfp = f"{HF}.layers.{i}.self_attn.indexer"
                tgt = f"model.layers.{i}.self_attn.indexer"
                layer_map[f"{hfp}.index_kpool_compress_gate.weight"] = WeightMapping(
                    target_path=f"{tgt}.kpool_gate.weight", sharding=(None, None), transpose=True
                )
                layer_map[f"{hfp}.index_kpool_compress_ape"] = WeightMapping(
                    target_path=f"{tgt}.kpool_ape", sharding=(None, None), transpose=False
                )
            if not cfg.is_kda_layer(i):
                layer_map = self._fix_ignored_dsa_quant(i, layer_map, model_config)
            mappings.update(layer_map)
            mappings.update(self._create_hc_mappings(i, cfg, HF))
        return mappings

    @staticmethod
    def _fix_ignored_dsa_quant(i: int, layer_map: dict, model_config) -> dict:
        """GLM-5.2's builder emits `.weight_q` + `.weight_scale` for every linear when the checkpoint
        is static-FP8. GLM-5.3's checkpoint leaves several attention tensors in bf16
        (`kv_b_proj`, norms, indexer, ...) per `modules_to_not_convert`. For those, the target must be
        the plain `.weight` with no scale sidecar. Detect via the lack of a `_scale_inv` sibling in the
        checkpoint's tensor set and downgrade the mapping."""
        from sgl_jax.srt.utils.weight_utils import WeightMapping

        qc = getattr(model_config, "quantization_config", None)
        ignored = getattr(qc, "ignored_layers", None) if qc is not None else None
        if not ignored:
            return layer_map

        import re

        def _canon(p: str) -> str:
            p = p.replace("/", ".")
            p = re.sub(r"\.(\d+)\.", r"[\1].", p)
            p = p.replace("language_model.", "")
            p = p.replace(".attn.", ".")
            p = re.sub(r"\.hc_(attn|ffn)_(fn|base|scale)", r".hc_\1.\2", p)
            return p

        canon_ignored = [_canon(ig) for ig in ignored]

        def is_ignored(module_path: str) -> bool:
            c = _canon(module_path)
            return any(c == ig or c.endswith(f".{ig}") for ig in canon_ignored)

        out = {}
        ignored_modules = set()
        for hf_key, mapping in layer_map.items():
            if not isinstance(mapping, WeightMapping):
                continue
            tp = mapping.target_path
            if isinstance(tp, str) and tp.endswith("weight_q"):
                module_path = tp[: -len(".weight_q")]
                if is_ignored(module_path):
                    ignored_modules.add(module_path)
        for hf_key, mapping in layer_map.items():
            if not isinstance(mapping, WeightMapping):
                out[hf_key] = mapping
                continue
            tp = mapping.target_path
            if not isinstance(tp, str) or not tp.endswith("weight_q"):
                # drop scale sidecars for ignored modules
                if isinstance(tp, str) and tp.endswith(".weight_scale"):
                    mod = tp[: -len(".weight_scale")]
                    if mod in ignored_modules:
                        continue
                out[hf_key] = mapping
                continue
            module_path = tp[: -len(".weight_q")]
            if module_path in ignored_modules:
                sh = mapping.sharding
                fixed_sh = (sh[1], sh[0]) if (sh and len(sh) == 2) else sh
                out[hf_key] = WeightMapping(
                    target_path=module_path + ".weight", sharding=fixed_sh, transpose=True
                )
            else:
                out[hf_key] = mapping
        return out

    @staticmethod
    def _create_hc_mappings(i: int, cfg: Glm5NextConfig, HF: str = "model.layers") -> dict:
        """mHC tensors: hc_attn_{fn,base,scale}, hc_ffn_{fn,base,scale}."""
        if not (cfg.hc_mult and cfg.hc_mult > 1):
            return {}
        p = f"{HF}.layers.{i}"
        out: dict = {}
        for site in ("attn", "ffn"):
            out[f"{p}.hc_{site}_fn"] = WeightMapping(
                target_path=f"model.layers.{i}.hc_{site}.fn", sharding=(None, None), transpose=True
            )
            out[f"{p}.hc_{site}_base"] = WeightMapping(
                target_path=f"model.layers.{i}.hc_{site}.base", sharding=(None,), transpose=False
            )
            out[f"{p}.hc_{site}_scale"] = WeightMapping(
                target_path=f"model.layers.{i}.hc_{site}.scale", sharding=(None,), transpose=False
            )
        return out

    @staticmethod
    def _create_kda_attention_mappings(i: int, cfg: Glm5NextConfig, HF: str = "model.layers") -> dict:
        """KDA (`KimiDeltaAttention`) projections only (norms are added by the base builder)."""
        prefix = f"{HF}.layers.{i}"
        tgt = f"model.layers.{i}.self_attn"
        la = cfg.linear_attn_config
        conv_size = la["short_conv_kernel_size"]
        projection_size = la["num_heads"] * la["head_dim"]

        out: dict = {}
        for name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "b_proj", "g_b_proj"):
            out[f"{prefix}.self_attn.{name}.weight"] = WeightMapping(
                target_path=f"{tgt}.{name}.weight", sharding=(None, "tensor"), transpose=True
            )
        for name in ("f_a_proj", "g_a_proj"):
            out[f"{prefix}.self_attn.{name}.weight"] = WeightMapping(
                target_path=f"{tgt}.{name}.weight", sharding=(None, None), transpose=True
            )
        out[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
            target_path=f"{tgt}.o_proj.weight", sharding=("tensor", None), transpose=True
        )
        for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
            out[f"{prefix}.self_attn.{conv}.weight"] = WeightMapping(
                target_path=f"{tgt}.attn.{conv}.weight",
                sharding=("tensor", None),
                transpose=False,
                reshape=(projection_size, conv_size),
            )
        out[f"{prefix}.self_attn.o_norm.weight"] = WeightMapping(
            target_path=f"{tgt}.o_norm.weight", sharding=(None,), transpose=False
        )
        out[f"{prefix}.self_attn.dt_bias"] = WeightMapping(
            target_path=f"{tgt}.attn.dt_bias", sharding=("tensor",), transpose=False
        )
        out[f"{prefix}.self_attn.A_log"] = WeightMapping(
            target_path=f"{tgt}.A_log", sharding=(None, None, "tensor", None), transpose=False,
            # Checkpoint stores A_log as 1-D [num_heads]; the param is [1, 1, H, 1] (Kimi layout).
            reshape=(1, 1, cfg.linear_attn_config["num_heads"], 1),
        )
        return out

    def get_input_embeddings(self):
        return self.model.embed_tokens


# Alias for readability; the registry keys on the class name, which must match the HF arch.
Glm5NextForCausalLM = Glm5NextForConditionalGeneration

EntryClass = [Glm5NextForConditionalGeneration]