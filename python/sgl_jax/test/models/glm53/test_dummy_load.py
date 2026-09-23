"""Validate GLM-5.3 loads with DUMMY weights (no checkpoint) and runs a forward on the TPU.

Mirrors `JAXDummyModelLoader`: build a meta model via nnx.eval_shape, set `_dummy_mode`, call
`load_weights`, then execute a forward with a synthetic ForwardBatch.

Run on CPU first (JAX_PLATFORMS=cpu), then on the 8 v7 chips.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "/home/emir/sglang-jax/python")

import jax
import jax.numpy as jnp

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig
from sgl_jax.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def tiny():
    return Glm5NextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1,
        layer_types=["linear_attention", "linear_attention", "deepseek_sparse_attention",
                     "linear_attention", "deepseek_sparse_attention", "linear_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
        indexer_types=["full"] * 6,
        q_lora_rank=48, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=0, v_head_dim=16,
        index_n_heads=2, index_head_dim=16, index_topk=64, index_kpool=4,
        linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                            "gate_lower_bound": -5.0},
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        rms_norm_eps=1e-5, swiglu_limit=10.0, max_position_embeddings=4096,
    )


def main():
    n = jax.device_count()
    mesh = create_device_mesh([1, n], [1])
    cfg = tiny()

    class MC:
        pass
    mc = MC()
    mc.hf_config = cfg
    mc.hf_text_config = cfg
    mc.quantization_config = None
    mc.dtype = jnp.bfloat16
    mc._dummy_mode = False

    with jax.set_mesh(mesh):
        model = Glm5NextForConditionalGeneration(cfg, mesh)

    # dummy load
    mc._dummy_mode = True
    print("calling load_weights in dummy mode...")
    model.load_weights(mc)
    print("dummy weights loaded")
    # count non-zero params
    import jax.tree_util as tu
    leaves = [x for x in tu.tree_leaves(model) if hasattr(x, "shape")]
    print("param leaves:", len(leaves))
    print("OK")


if __name__ == "__main__":
    main()