"""Construct a tiny Glm5Next model on CPU to validate the module graph and shapes.

No forward pass yet — this catches construction/attribute errors cheaply.
"""
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

sys.path.insert(0, "/home/emir/sglang-jax/python")

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig
from sgl_jax.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def make_mesh(n=1):
    return create_device_mesh([1, n], [1])


def tiny():
    return Glm5NextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        layer_types=[
            "linear_attention", "linear_attention", "deepseek_sparse_attention",
            "linear_attention", "deepseek_sparse_attention", "linear_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse", "sparse", "sparse"],
        indexer_types=["full"] * 6,
        q_lora_rank=48, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=0, v_head_dim=16,
        index_n_heads=2, index_head_dim=64, index_topk=64, index_kpool=4,
        linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                            "gate_lower_bound": -5.0},
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        rms_norm_eps=1e-5, swiglu_limit=10.0, max_position_embeddings=4096,
    )


def main():
    cfg = tiny()
    print("config:", type(cfg).__name__, "kda layers", cfg.linear_layer_ids, "full", cfg.full_attention_layer_ids)
    mesh = create_device_mesh([1, jax.device_count()], [1])
    with jax.set_mesh(mesh):
        model = Glm5NextForConditionalGeneration(cfg, mesh)
    print("model constructed OK")
    # count params
    import numpy as np
    n = 0
    def walk(m):
        nonlocal n
        if hasattr(m, "value") and hasattr(m.value, "shape"):
            n += int(np.prod(m.value.shape))
    jax.tree.map(walk, jax.tree_util.tree_leaves_with_path(model))
    print("constructed. module tree leaves:", len(jax.tree_util.tree_leaves(model)))
    print("OK")


if __name__ == "__main__":
    main()