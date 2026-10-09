"""GLM-5.3 mHC Pallas kernels vs an f32 XLA reference of the ``Glm5NextHC`` math.

JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python -m pytest -q python/sgl_jax/test/test_mhc_kernel.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.mhc.glm5_mhc import glm5_mhc_post, glm5_mhc_pre, unpack_state
from sgl_jax.srt.models.glm5_next import Glm5NextHC

HC, ITERS, HC_EPS, RMS_EPS = 4, 20, 1e-6, 1e-5
# Native Pallas on TPU, interpret mode elsewhere.
INTERPRET = jax.default_backend() != "tpu"


def _ref_pre(streams, fn, scale, base):
    """f32 XLA reference (HIGHEST precision): -> (collapsed, post, comb). fn is [hc*D, n]."""
    hc = HC
    lead = streams.shape[:-1]
    d = streams.shape[-1] // hc
    flat = streams.astype(jnp.float32)
    flat = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + RMS_EPS)
    mix = jnp.dot(flat, fn.astype(jnp.float32), precision="highest")
    pre_w, post_w, comb_w = mix[..., :hc], mix[..., hc : 2 * hc], mix[..., 2 * hc :]
    pre = jax.nn.sigmoid(pre_w * scale[0] + base[:hc]) + HC_EPS
    post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + base[hc : 2 * hc])
    logits = comb_w.reshape(*lead, hc, hc) * scale[2] + base[2 * hc :].reshape(hc, hc)
    comb = jax.nn.softmax(logits, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    s = streams.reshape(*lead, hc, d).astype(jnp.float32)
    collapsed = jnp.einsum("...h,...hd->...d", pre, s, precision="highest")
    return collapsed.astype(streams.dtype), post, comb


def _ref_post(y, post, comb, streams):
    lead = streams.shape[:-1]
    d = streams.shape[-1] // HC
    s = streams.reshape(*lead, HC, d).astype(jnp.float32)
    mixed = jnp.einsum("...ij,...id->...jd", comb, s, precision="highest")
    out = post[..., None] * y.astype(jnp.float32)[..., None, :] + mixed
    return out.reshape(*lead, HC * d).astype(streams.dtype)


def _make(t, d, seed, f32=False):
    """Random inputs + a Glm5NextHC loaded through its weight-loading recipe."""
    n = (2 + HC) * HC
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    dt = jnp.float32 if f32 else jnp.bfloat16
    streams = (jax.random.normal(ks[0], (t, HC * d)) * 1.5).astype(dt)
    y = jax.random.normal(ks[1], (t, d)).astype(dt)
    fn = (jax.random.normal(ks[2], (HC * d, n)) * (2.0 / np.sqrt(HC * d))).astype(dt)
    base = jax.random.normal(ks[3], (n,)).astype(jnp.float32)
    scale = jax.random.uniform(ks[4], (3,), minval=0.5, maxval=2.0).astype(jnp.float32)
    mesh = jax.make_mesh((1, 1), ("data", "tensor"), devices=jax.devices()[:1])
    hcm = Glm5NextHC(d, HC, ITERS, HC_EPS, RMS_EPS, mesh, dt)
    # Checkpoint layout: fn is [n, hc*D].
    outputs = hcm._prepare_weights([np.asarray(fn.T), np.asarray(base), np.asarray(scale)])
    for param, value in zip((hcm.fn_t, hcm.scale_rows, hcm.base_rows), outputs):
        param[...] = value
    return hcm, streams, y, (fn, scale, base)


def _run_kernels(hcm, streams, y, block_tokens=None):
    """The production kernel calls on the load-time prepared weights (as plain arrays: the params
    carry explicit mesh shardings, which Glm5NextHC strips with shard_map)."""
    prepared = tuple(
        jnp.asarray(np.asarray(p[...])) for p in (hcm.fn_t, hcm.scale_rows, hcm.base_rows)
    )
    collapsed, state = glm5_mhc_pre(
        streams,
        *prepared,
        hc=HC,
        hc_iters=ITERS,
        hc_eps=HC_EPS,
        eps=RMS_EPS,
        block_tokens=block_tokens,
        interpret=INTERPRET,
    )
    out = glm5_mhc_post(y, state, streams, hc=HC, block_tokens=block_tokens, interpret=INTERPRET)
    return collapsed, state, out


def _check_against_reference(hcm, streams, y, weights, block_tokens=None):
    collapsed_ref, post_ref, comb_ref = _ref_pre(streams, *weights)
    out_ref = _ref_post(y, post_ref, comb_ref, streams)
    collapsed, state, out = _run_kernels(hcm, streams, y, block_tokens)
    _, post, comb = unpack_state(state, HC)

    assert collapsed.shape == collapsed_ref.shape and collapsed.dtype == streams.dtype
    assert out.shape == streams.shape and out.dtype == streams.dtype
    assert np.isfinite(np.asarray(state)).all()
    np.testing.assert_allclose(np.asarray(post), np.asarray(post_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(comb), np.asarray(comb_ref), rtol=1e-4, atol=1e-5)
    # bf16 outputs: allow ~1 bf16 ulp from differing f32 summation order before rounding.
    f = lambda a: np.asarray(a.astype(jnp.float32))
    np.testing.assert_allclose(f(collapsed), f(collapsed_ref), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(f(out), f(out_ref), rtol=1e-2, atol=2e-2)
    return state


def _np_sinkhorn(logits, iters, eps):
    """Independent float64 reference of the exact variant used by glm53 (eps placement)."""
    c = np.exp(logits - logits.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + eps
    c = c / (c.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdims=True) + eps)
        c = c / (c.sum(-2, keepdims=True) + eps)
    return c


# Default tiling: bt=32 for T <= 256 (T=100: partial tail), bt=128 above (T=300: partial tail).
@pytest.mark.parametrize(
    "t,d", [(4, 4096), (16, 4096), (64, 4096), (100, 4096), (300, 256), (512, 1024)]
)
def test_mhc_kernel_matches_reference(t, d):
    hcm, streams, y, weights = _make(t, d, seed=t)
    state = _check_against_reference(hcm, streams, y, weights)
    pre, _, comb = unpack_state(state, HC)

    # Gates against an independent float64 recomputation (exact eps placement).
    fn, scale, base = weights
    flat = np.asarray(streams.astype(jnp.float32), np.float64)
    flat = flat / np.sqrt((flat * flat).mean(-1, keepdims=True) + RMS_EPS)
    mix = flat @ np.asarray(fn.astype(jnp.float32), np.float64)
    sc, bs = np.asarray(scale, np.float64), np.asarray(base, np.float64)
    pre64 = 1 / (1 + np.exp(-(mix[:, :HC] * sc[0] + bs[:HC]))) + HC_EPS
    np.testing.assert_allclose(np.asarray(pre), pre64, rtol=1e-4, atol=1e-5)
    logits = mix[:, 2 * HC :].reshape(t, HC, HC) * sc[2] + bs[2 * HC :].reshape(HC, HC)
    np.testing.assert_allclose(
        np.asarray(comb), _np_sinkhorn(logits, ITERS, HC_EPS), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("bt", [16, 32])
def test_mhc_kernel_block_sizes_partial_tail(bt):
    t, d = 72, 512  # 72 % 32 != 0 -> partial edge block
    hcm, streams, y, weights = _make(t, d, seed=7)
    _check_against_reference(hcm, streams, y, weights, block_tokens=bt)


def test_mhc_kernel_f32_inputs():
    t, d = 24, 256
    hcm, streams, y, weights = _make(t, d, seed=3, f32=True)
    collapsed_ref, post_ref, comb_ref = _ref_pre(streams, *weights)
    out_ref = _ref_post(y, post_ref, comb_ref, streams)
    collapsed, _, out = _run_kernels(hcm, streams, y, block_tokens=16)
    np.testing.assert_allclose(
        np.asarray(collapsed), np.asarray(collapsed_ref), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=1e-4, atol=1e-4)


def test_glm5_next_hc_module(monkeypatch):
    """Glm5NextHC.pre/post under a ('data', 'tensor') mesh + jit; with >= 4 devices (e.g.
    JAX_NUM_CPU_DEVICES=4) tokens are sharded over 'data' as under DP attention."""
    if INTERPRET:  # the model leaves `interpret` to glm5_mhc's get_interpret() default
        monkeypatch.setenv("PALLAS_INTERPRET", "1")
    t, d = 32, 512
    hcm, streams, y, weights = _make(t, d, seed=11)
    ref_c, ref_post, ref_comb = _ref_pre(streams, *weights)
    ref_out = _ref_post(y, ref_post, ref_comb, streams)

    n_dev = len(jax.devices())
    dp = tp = 2 if n_dev >= 4 else 1
    mesh = jax.make_mesh((dp, tp), ("data", "tensor"), devices=jax.devices()[: dp * tp])
    hcm.mesh = mesh
    if dp > 1:  # DP attention: tokens sharded over 'data', features replicated
        streams = jax.device_put(streams, jax.NamedSharding(mesh, jax.P("data", None)))
        y = jax.device_put(y, jax.NamedSharding(mesh, jax.P("data", None)))

    @jax.jit
    def run(s, yy):
        c, st = hcm.pre(s)
        return c, hcm.post(yy, st, s)

    c, out = run(streams, y)
    f = lambda a: np.asarray(a.astype(jnp.float32))
    np.testing.assert_allclose(f(c), f(ref_c), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(f(out), f(ref_out), rtol=1e-2, atol=2e-2)
