"""Fast CPU component tests for GLM-5.3 attention modules — no TPU server needed.

Calls the KDA module and the DSA (MLA+indexer) module directly with a stubbed attention backend,
so projection/rope/conv/indexer shape bugs surface in seconds.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/emir/sglang-jax/python")

import jax
import jax.numpy as jnp

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig
from sgl_jax.srt.models.glm5_next import Glm5NextHC, Glm5NextDecoderLayer
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

T = 13


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
        index_n_heads=2, index_head_dim=64, index_topk=64, index_kpool=4,
        linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                            "gate_lower_bound": -5.0},
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        rms_norm_eps=1e-5, swiglu_limit=10.0, max_position_embeddings=4096,
    )


class StubBackend:
    def __init__(self):
        self.calls = []
    def __call__(self, *args, **kw):
        self.calls.append(dict(nargs=len(args), kw=list(kw)))
        layer = kw.get("layer")
        if layer is not None and hasattr(layer, "num_q_heads"):
            return jnp.zeros((args[0].shape[0], layer.num_q_heads, layer.head_v_dim), jnp.float32), "ST"
        # MLA absorbed path: attn_output is the latent output [T, H, kv_lora_rank]
        q = args[0]
        return jnp.zeros((q.shape[0], q.shape[1], 32), jnp.float32), "ST"


def main():
    mesh = create_device_mesh([1, 1], [1])
    cfg = tiny()
    with jax.set_mesh(mesh):
        # ---- mHC ----
        hc = Glm5NextHC(64, 4, 20, 1e-6, mesh, jnp.bfloat16)
        x = jnp.ones((T, 256), jnp.bfloat16)
        c, st = hc.pre(x); o = hc.post(jnp.ones_like(c), st, x)
        print("[mHC] pre", c.shape, "post", o.shape)

        # ---- KDA attention module (layer 0) ----
        l0 = Glm5NextDecoderLayer(cfg, mesh, layer_id=0, dtype=jnp.bfloat16)
        be = StubBackend()
        fb = SimpleNamespace(attn_backend=be)
        h = jnp.ones((T, 64), jnp.bfloat16)
        try:
            kda_out, kda_state = l0.self_attn(fb.positions if hasattr(fb,'positions') else None, h, fb, "REC_POOL")
            print("[KDA] out", kda_out.shape, "state", kda_state)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("[KDA] FAILED:", type(e).__name__, str(e)[:200])

        # ---- DSA attention module (layer 2) ----
        l2 = Glm5NextDecoderLayer(cfg, mesh, layer_id=2, dtype=jnp.bfloat16)
        be2 = StubBackend()
        fb2 = SimpleNamespace(attn_backend=be2)
        pos = jnp.arange(T)
        try:
            dsa_out, dsa_state = l2.self_attn(pos, h, fb2, "KV_POOL")
            print("[DSA] out", dsa_out.shape, "state", dsa_state)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("[DSA] FAILED:", type(e).__name__, str(e)[:200])
    print("DONE")


if __name__ == "__main__":
    main()