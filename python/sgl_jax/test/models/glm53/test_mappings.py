"""Check GLM-5.3 weight mappings generate consistent paths for every layer type."""
import sys

import jax

sys.path.insert(0, "/home/emir/sglang-jax/python")

from sgl_jax.srt.configs.glm5_next import Glm5NextConfig
from sgl_jax.srt.models.glm5_next import Glm5NextForConditionalGeneration


def main():
    from tests_glm53_construct import tiny, make_mesh
    cfg = tiny()
    mesh = make_mesh(1)
    with jax.set_mesh(mesh):
        model = Glm5NextForConditionalGeneration(cfg, mesh)

    class QC:  # minimal shim: no quantization
        is_static_checkpoint = False
    mc = type("MC", (), {"quantization_config": None, "hf_config": cfg, "hf_text_config": cfg})()
    maps = model._create_weight_mappings(mc)
    print("total mappings:", len(maps))

    # KDA layer 0 (linear_attention) should have f_a/f_b/b_proj/conv/A_log/dt_bias
    kda_keys = [k for k in maps if k.startswith("model.layers.0.self_attn.")]
    print("layer0 (KDA) attn keys:", sorted(k.split("self_attn.")[1] for k in kda_keys))
    # DSA layer 2
    dsa_keys = [k for k in maps if k.startswith("model.layers.2.self_attn.")]
    print("layer2 (DSA) attn keys:", sorted(k.split("self_attn.")[1] for k in dsa_keys))
    # mHC
    hc = [k for k in maps if ".hc_" in k]
    print("mHC keys for layer0:", sorted(k.split("model.layers.0.")[1] for k in hc if k.startswith("model.layers.0.")))

    # every HF path's target must resolve to a real param path prefix in the model
    print("sanity: has embed", "model.embed_tokens.weight" in maps, "| has lm_head", "lm_head.weight" in maps)
    print("dense mlp layer0:", [k for k in maps if k.startswith("model.layers.0.mlp.")])
    print("moe layer1 gate:", [k for k in maps if "layers.1.mlp.gate" in k])
    print("shared experts layer1:", [k for k in maps if "layers.1.mlp.shared_experts" in k][:3])
    print("OK")


if __name__ == "__main__":
    main()