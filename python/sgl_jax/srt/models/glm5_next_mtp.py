"""GLM-5.3-Flash (`glm5_next`) multi-token-prediction (NextN / MTP) draft model.

The checkpoint ships one MTP block as decoder layer ``num_hidden_layers`` (45):
``eh_proj``/``enorm``/``hnorm``/``shared_head.norm`` plus an ordinary DSA (NoPE MLA +
lightning indexer) + MoE decoder layer. Unlike the target's decoder layers it has no
mHC tensors and no KDA, so it is a plain pre-norm residual block. Embedding and
``lm_head`` are shared with the target (set via ``set_embed_and_head``).

Forward (mirrors upstream sglang ``DeepseekV3ForCausalLMNextN``):

    x = eh_proj(concat(enorm(embed(input_ids)), hnorm(h_prev)))
    x = decoder_layer(x)                        # DSA attention + MoE, additive residual
    h = shared_head.norm(x)                     # returned as the next step's h_prev
    logits = lm_head(h)

``h_prev`` is the target's post-final-norm hidden (mHC streams mean-collapsed) on the
first step and this model's own ``h`` on later draft steps.

``ModelConfig._patch_glm5_next_nextn_draft_config`` turns the target config into a
1-layer DSA-only, all-MoE config, so the decoder layer is built as layer 0 and the
KV / indexer pools are sized for one layer.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig, get_glm5_next_config
from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM
from sgl_jax.srt.models.glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextForConditionalGeneration,
)

logger = logging.getLogger(__name__)


class Glm5NextMTPDecoderLayer(Glm5NextDecoderLayer):
    """The target's DSA + MoE decoder layer as a plain pre-norm residual block (no mHC)."""

    use_mhc = False

    def __call__(
        self, positions, hidden_states, forward_batch, token_to_kv_pool, dispatch_info=None
    ):
        h = self.input_layernorm(hidden_states)
        attn_out, attn_state = self.self_attn(positions, h, forward_batch, token_to_kv_pool)
        hidden_states = hidden_states + attn_out
        h = self.post_attention_layernorm(hidden_states)
        router_logits = self.moe_gate(h)
        topk_weights, topk_ids = self.topk(
            router_logits, self.moe_gate.bias.value, dispatch_info=dispatch_info
        )
        limit = self.swiglu_limit
        mlp_out = self.mlp(h, topk_weights, topk_ids, swiglu_limit=limit, shared_swiglu_limit=limit)
        return hidden_states + mlp_out, attn_state


class Glm5NextMTPDraftModel(nnx.Module):
    def __init__(self, config: Glm5NextConfig, mesh, dtype=jnp.bfloat16):
        hidden = config.hidden_size
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=hidden,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.enorm = RMSNorm(
            hidden, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="enorm"
        )
        self.hnorm = RMSNorm(
            hidden, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="hnorm"
        )
        self.eh_proj = LinearBase(
            input_size=2 * hidden,
            output_size=hidden,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [Glm5NextMTPDecoderLayer(config=config, mesh=mesh, layer_id=0, dtype=dtype)]
        )
        # checkpoint `shared_head.norm`
        self.norm = RMSNorm(
            hidden, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool):
        embed = self.embed_tokens(forward_batch.input_ids)
        hidden_in = forward_batch.spec_info.hidden_states
        emb_sh = jax.typeof(embed).sharding
        if isinstance(emb_sh, jax.sharding.NamedSharding):
            hidden_in = jax.sharding.reshard(hidden_in, emb_sh)
        hidden_in = hidden_in.astype(embed.dtype)
        hidden_states, _ = self.eh_proj(
            jnp.concatenate((self.enorm(embed), self.hnorm(hidden_in)), axis=-1)
        )
        hidden_states, attn_state = self.layers[0](
            forward_batch.positions,
            hidden_states,
            forward_batch,
            token_to_kv_pool,
            dispatch_info=forward_batch.expert_location_metadata,
        )
        hidden_states = self.norm(hidden_states)
        # DSAFusedCache (dsa_sparse is required).
        kv_update = (
            ([attn_state.kv], [attn_state.idx]) if attn_state.idx is not None else [attn_state.kv]
        )
        return hidden_states, kv_update


class Glm5NextForCausalLMNextN(nnx.Module):
    """Single-block MTP draft for GLM-5.3-Flash (arch set by the draft ModelConfig)."""

    load_lm_head_from_target = True

    @classmethod
    def patch_model_config(cls, config: ModelConfig) -> None:
        Glm5NextForConditionalGeneration.patch_model_config(config)

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        text = getattr(config, "text_config", config)
        cfg = get_glm5_next_config(config, force=True)
        if getattr(config, "quantization_config", None) is not None:
            cfg.quantization_config = config.quantization_config
        cfg.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
        if cfg.moe_backend != MoEBackend.FUSED_V2:
            raise ValueError("GLM-5.3-Flash currently supports only --moe-backend fused_v2")
        cfg.ep_size = getattr(config, "ep_size", 1)
        self.mtp_layer_id = int(text._sgl_mtp_layer_id)
        self.config = cfg
        self.mesh = mesh
        self.dtype = dtype
        self.model = Glm5NextMTPDraftModel(cfg, mesh=mesh, dtype=dtype)
        self.lm_head = ParallelLMHead(
            cfg.vocab_size, cfg.hidden_size, dtype=dtype, param_dtype=dtype, mesh=mesh
        )
        self.logits_processor = LogitsProcessor(cfg.vocab_size, mesh=mesh)
        self.hot_token_ids = None

    def get_shared_weight_paths(self):
        return ("model.embed_tokens.embedding", "lm_head.embedding")

    def __call__(self, forward_batch: ForwardBatch, memory_pools, logits_metadata: LogitsMetadata):
        hidden, kv_update = self.model(forward_batch, memory_pools.token_to_kv_pool)
        output = self.logits_processor(hidden, self.lm_head, logits_metadata)
        return output, {"token_to_kv_pool": kv_update}, True, None

    # ---- embed / head sharing with the target ----
    def get_embed_and_head(self):
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(self, embed: jax.Array, head: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed
        self.lm_head.embedding.value = head

    def set_embed(self, embed: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed

    # ---- weights ----
    def load_weights(self, model_config: ModelConfig):
        # Absorbed-MLA preparation runs as Glm5Attention's prepare_weight_loading hook.
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load(self._create_weight_mappings(model_config))
        logger.info(
            "GLM-5.3-Flash MTP draft weights loaded (checkpoint layer %d).", self.mtp_layer_id
        )

    def prepare_weight_loading(self, loader, mappings):
        # Block-wise FP8 shared experts are requantized per channel at load, as for the target.
        return Glm5ForCausalLM.prepare_weight_loading(self, loader, mappings)

    def _create_weight_mappings(self, model_config: ModelConfig) -> dict:
        HF = "model.language_model"
        src = f"{HF}.layers.{self.mtp_layer_id}"
        quant_config = model_config.quantization_config
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint
        # The draft ModelConfig renames the MTP layer's BF16 modules to the draft module tree.
        unquantized = (
            frozenset(quant_config.ignored_layers or []) if is_static_quant else frozenset()
        )
        mappings: dict = {
            f"{src}.eh_proj.weight": WeightSpec(
                target_path="model.eh_proj.weight", sharding=(None, None), transpose=True
            ),
            f"{src}.enorm.weight": WeightSpec(
                target_path="model.enorm.scale", sharding=(None,), transpose=False
            ),
            f"{src}.hnorm.weight": WeightSpec(
                target_path="model.hnorm.scale", sharding=(None,), transpose=False
            ),
            f"{src}.shared_head.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        # Draft layer 0 <- checkpoint layer `mtp_layer_id` (DSA attention + MoE).
        mappings.update(
            Glm5ForCausalLM._create_moe_layer_mappings(
                self,
                0,
                self.mtp_layer_id,
                False,
                is_static_quant,
                True,
                hf_prefix=f"{HF}.layers",
                unquantized=unquantized,
            )
        )
        hfp = f"{src}.self_attn.indexer"
        tgt = "model.layers.0.self_attn.indexer"
        mappings[f"{hfp}.index_kpool_compress_gate"] = WeightSpec(
            target_path=f"{tgt}.index_kpool_compress_gate.weight",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{hfp}.index_kpool_compress_ape"] = WeightSpec(
            target_path=f"{tgt}.kpool_ape", sharding=(None, None), transpose=False
        )
        return mappings


EntryClass = [Glm5NextForCausalLMNextN]
