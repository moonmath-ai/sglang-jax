"""Grouped-matmul prefill EP-MoE vs the fused kernel's reference (``ref_moe``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v2.kernel import ref_moe
from sgl_jax.srt.layers.moe_gmm_prefill import gmm_prefill_moe

E, H, INTER, IS, T, K, QB, LIMIT = 16, 512, 256, 256, 64, 4, 128, 10.0


@pytest.mark.parametrize("act_quant,tol", [(False, 1e-2), (True, 3e-2)])
@pytest.mark.parametrize("scatter", [False, True])
def test_gmm_prefill_moe_matches_reference(act_quant, tol, scatter):
    n = max(d for d in (8, 4, 2, 1) if d <= len(jax.devices()) and E % d == 0)
    mesh = jax.make_mesh((1, n), ("data", "tensor"), devices=jax.devices()[:n])
    exp = lambda nd: NamedSharding(mesh, P(("data", "tensor"), *([None] * (nd - 1))))
    rep = NamedSharding(mesh, P())
    rng = np.random.default_rng(0)
    f8 = lambda *s: jnp.asarray(rng.normal(0, 1, s), jnp.float32).astype(jnp.float8_e4m3fn)
    sc = lambda *s: jnp.asarray(rng.uniform(0.005, 0.02, s), jnp.float32)
    w1, w3, w2 = f8(E, H, INTER), f8(E, H, INTER), f8(E, INTER, H)
    s1, s3, s2 = sc(E, H // QB, 1, INTER), sc(E, H // QB, 1, INTER), sc(E, INTER // QB, 1, H)
    w1s, w3s, w2s = f8(H, IS), f8(H, IS), f8(IS, H)
    s1s, s3s, s2s = sc(1, 1, IS), sc(1, 1, IS), sc(1, 1, H)
    x = jnp.asarray(rng.normal(0, 1, (T, H)), jnp.bfloat16)
    ids = jnp.asarray(np.argsort(-rng.normal(size=(T, E)), axis=1)[:, :K], jnp.int32)
    tw = jnp.asarray(rng.uniform(0.05, 0.3, (T, K)), jnp.float32)

    ref = ref_moe(
        x,
        w1,
        w2,
        w3,
        np.asarray(tw),
        np.asarray(ids),
        K,
        act_fn="silu",
        swiglu_limit=LIMIT,
        shared_swiglu_limit=LIMIT,
        w1_shared=w1s,
        w2_shared=w2s,
        w3_shared=w3s,
        w1_shared_scale=s1s[:, 0, :],
        w2_shared_scale=s2s[:, 0, :],
        w3_shared_scale=s3s[:, 0, :],
        quant_block_k=QB,
        w1_scale=s1,
        w2_scale=s2,
        w3_scale=s3,
    )
    put = jax.device_put
    out = jax.jit(
        lambda *a: gmm_prefill_moe(
            mesh,
            *a,
            top_k=K,
            swiglu_limit=LIMIT,
            shared_swiglu_limit=LIMIT,
            act_quant=act_quant,
            scatter_tokens=scatter,
        )
    )(
        put(x, rep),
        put(tw, rep),
        put(ids, rep),
        put(w1, exp(3)),
        put(w3, exp(3)),
        put(w2, exp(3)),
        put(s1, exp(4)),
        put(s3, exp(4)),
        put(s2, exp(4)),
        put(w1s, rep),
        put(w3s, rep),
        put(w2s, rep),
        put(s1s, rep),
        put(s3s, rep),
        put(s2s, rep),
    )
    if scatter:  # token-sharded over the mesh
        assert out.sharding.spec[0] == ("data", "tensor"), out.sharding.spec
    ref = np.asarray(ref, np.float32)
    err = np.abs(np.asarray(out, np.float32) - ref).max() / np.abs(ref).max()
    assert err < tol, err
