"""Config for GLM-5.3-Flash (`glm5_next`) and relatives.

GLM-5.3-Flash is a hybrid model: 45 layers of `[3x KDA linear attention, 1x NoPE sparse MLA]`, a 288-expert
MoE (top-8 + 1 shared), four-stream manifold-constrained hyper-connections (mHC), and a K-pool-indexed DSA
sparse attention. It reuses sglang-jax's KDA backend and DSA sparse backend.

This config is deliberately shaped like `BailingHybridConfig` / `KimiLinearConfig` so the existing hybrid
plumbing (`attn_backend_windows`, `RecurrentStatePool`, `HybridLinearKVPool`) works unchanged:

  * `use_kda` / `linear_layer_ids` / `full_attention_layer_ids` / `is_kda_layer`
  * `linear_state_params` (drives `RecurrentStatePool`)
  * `linear_attn_config` (read by `KDAAttnBackend` via the layer, not directly)

Detection: `glm5_next` model_type, or an architecture beginning with `Glm5Next.

Config facts (zai-org/GLM-5.3-Flash):
  hidden 4096, 45 layers, first_k_dense_replace 3, 288 experts top-8, moe_inter 2048,
  q_lora 1536, kv_lora 512, qk_nope 256, qk_rope 0 (NoPE), v_head 256, heads 64,
  KDA heads 64 head_dim 128 conv_k 4 gate_lower_bound -5.0,
  index heads 32 head_dim 128 topk 2048 kpool 4, hc_mult 4, swiglu_limit 10.0.
"""
from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_routed_experts: int = 288,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sigmoid",
        first_k_dense_replace: int = 3,
        rms_norm_eps: float = 1e-5,
        swiglu_limit: float = 10.0,
        # attention layer schedule
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        # MLA (NoPE)
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        mla_use_nope: bool = True,
        use_qk_norm: bool = True,
        # KDA
        linear_attn_config: dict[str, Any] | None = None,
        kda_lower_bound: float | None = -5.0,
        # DSA indexer / K-pool
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        index_kpool: int = 4,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        indexer_rope_interleave: bool = True,
        index_share_for_mtp_iteration: bool = True,
        # mHC
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # rope / position
        rope_theta: float = 800000.0,
        max_position_embeddings: int = 1048576,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        quantization_config: dict | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts_per_token = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.num_expert_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.moe_renormalize = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.score_function = scoring_func
        self.first_k_dense_replace = first_k_dense_replace
        self.rms_norm_eps = rms_norm_eps
        self.swiglu_limit = swiglu_limit

        # The transformers Glm5NextTextConfig stores these lists verbatim; keep them.
        self.layer_types = list(layer_types or [])
        self.mlp_layer_types = list(mlp_layer_types or [])
        self.indexer_types = list(indexer_types) if indexer_types is not None else None

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.use_qk_norm = use_qk_norm

        self.linear_attn_config = linear_attn_config or {
            "num_heads": num_attention_heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": kda_lower_bound,
        }
        # KDAAttnBackend reads a bound off the layer; `RadixLinearAttention` gets `kda_lower_bound`.
        self.kda_lower_bound = kda_lower_bound
        self.kda_safe_gate = kda_lower_bound is not None
        self.use_kda = True

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.indexer_rope_interleave = indexer_rope_interleave
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration

        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias

        if quantization_config is not None:
            self.quantization_config = quantization_config

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ── layer taxonomy (mirrors BailingHybridConfig) ──
    @property
    def _resolved_layer_types(self) -> list[str]:
        if self.layer_types:
            return self.layer_types
        # fall back to the [3 KDA, 1 DSA] rhythm
        return [
            "deepseek_sparse_attention" if (i % 4 == 3) else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]

    def is_kda_layer(self, layer_idx: int) -> bool:
        return str(self._resolved_layer_types[layer_idx]).lower() == "linear_attention"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return not self.is_kda_layer(layer_idx)

    @property
    def linear_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def linear_attn_layers(self) -> list[int]:
        """Alias used by the generic hybrid runner path."""
        return self.linear_layer_ids

    @property
    def full_attn_layers(self) -> list[int]:
        return self.full_attention_layer_ids

    @property
    def head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @head_dim.setter
    def head_dim(self, value: int) -> None:
        # ModelConfig assigns head_dim; keep the nope/rope split authoritative and ignore the flat value.
        if value is not None and value != self.qk_nope_head_dim + self.qk_rope_head_dim:
            self.qk_nope_head_dim = max(value - self.qk_rope_head_dim, self.qk_nope_head_dim)

    @property
    def linear_state_params(self):
        from sgl_jax.srt.mem_cache.recurrent_state_pool import (
            LinearRecurrentStateParams,
            recurrent_state_dtype,
        )

        la = self.linear_attn_config
        return LinearRecurrentStateParams(
            layers=self.linear_layer_ids,
            num_heads=la["num_heads"],
            head_dim=la["head_dim"],
            conv_kernel_size=la["short_conv_kernel_size"],
            dtype=recurrent_state_dtype(),
        )

    def is_mla(self) -> bool:
        return True


def _is_glm5_next_config(hf_config: Any) -> bool:
    model_type = str(getattr(hf_config, "model_type", "") or "")
    if model_type in ("glm5_next", "glm5_next_text"):
        return True
    if "Glm5Next" in type(hf_config).__name__:
        return True
    archs = getattr(hf_config, "architectures", None) or []
    return any(str(a).startswith("Glm5Next") for a in archs)


def get_glm5_next_config(hf_config: Any) -> Glm5NextConfig | None:
    """Return a normalized Glm5NextConfig for an HF glm5_next config, else None.

    The HF config may already be a `transformers.Glm5NextTextConfig` (nested under a
    top-level multimodal config), or a plain dict. We read the fields we need and build our own
    config type so the hybrid plumbing has a stable interface.
    """
    if not _is_glm5_next_config(hf_config):
        return None

    text = getattr(hf_config, "text_config", hf_config)

    def g(name: str, default=None):
        v = getattr(text, name, None)
        return default if v is None else v

    la = dict(getattr(text, "linear_attn_config", None) or {})
    if not la:
        la = {
            "num_heads": g("linear_num_heads", g("num_attention_heads", 64)),
            "head_dim": g("linear_head_dim", 128),
            "short_conv_kernel_size": g("linear_conv_kernel_dim", 4),
            "gate_lower_bound": g("linear_lower_bound", -5.0),
        }

    return Glm5NextConfig(
        vocab_size=g("vocab_size", 154880),
        hidden_size=g("hidden_size", 4096),
        intermediate_size=g("intermediate_size", 12288),
        moe_intermediate_size=g("moe_intermediate_size", 2048),
        num_hidden_layers=g("num_hidden_layers", 45),
        num_attention_heads=g("num_attention_heads", 64),
        num_key_value_heads=g("num_key_value_heads", g("num_attention_heads", 64)),
        n_routed_experts=g("n_routed_experts", 288),
        num_experts_per_tok=g("num_experts_per_tok", 8),
        n_shared_experts=g("n_shared_experts", 1),
        n_group=g("n_group", 1),
        topk_group=g("topk_group", 1),
        norm_topk_prob=g("norm_topk_prob", True),
        routed_scaling_factor=g("routed_scaling_factor", 2.5),
        scoring_func=g("scoring_func", "sigmoid"),
        first_k_dense_replace=g("first_k_dense_replace", 3),
        rms_norm_eps=g("rms_norm_eps", 1e-5),
        swiglu_limit=g("swiglu_limit", 10.0),
        layer_types=list(g("layer_types", []) or []),
        mlp_layer_types=list(g("mlp_layer_types", []) or []),
        indexer_types=(list(g("indexer_types")) if g("indexer_types") is not None else None),
        q_lora_rank=g("q_lora_rank", 1536),
        kv_lora_rank=g("kv_lora_rank", 512),
        qk_nope_head_dim=g("qk_nope_head_dim", 256),
        qk_rope_head_dim=g("qk_rope_head_dim", 0),
        v_head_dim=g("v_head_dim", 256),
        mla_use_nope=g("mla_use_nope", True),
        use_qk_norm=g("use_qk_norm", True),
        linear_attn_config=la,
        kda_lower_bound=la.get("gate_lower_bound", -5.0),
        index_n_heads=g("index_n_heads", 32),
        index_head_dim=g("index_head_dim", 128),
        index_topk=g("index_topk", 2048),
        index_kpool=g("index_kpool", 4),
        index_kpool_compress=g("index_kpool_compress", True),
        index_kpool_always_select_tail=g("index_kpool_always_select_tail", True),
        indexer_rope_interleave=g("indexer_rope_interleave", True),
        index_share_for_mtp_iteration=g("index_share_for_mtp_iteration", True),
        hc_mult=g("hc_mult", 4),
        hc_sinkhorn_iters=g("hc_sinkhorn_iters", 20),
        hc_eps=g("hc_eps", 1e-6),
        rope_theta=g("rope_theta", 800000.0),
        max_position_embeddings=g("max_position_embeddings", 1048576),
        tie_word_embeddings=g("tie_word_embeddings", False),
        attention_bias=g("attention_bias", False),
    )