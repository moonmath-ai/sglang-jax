"""GLM-5.3 mHC Pallas kernels vs the reference ``Glm5NextHC`` math (CPU: Pallas interpret mode).

    JAX_PLATFORMS=cpu python -m pytest -q python/sgl_jax/test/test_mhc_kernel.py
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.mhc.glm5_mhc import glm5_mhc_post, glm5_mhc_pre, unpack_state
from sgl_jax.srt.models import glm5_next as _glm5_next
from sgl_jax.srt.models.glm5_next import Glm5NextHC

HC, ITERS, HC_EPS, RMS_EPS = 4, 20, 1e-6, 1e-5
# Native Pallas on TPU (interpret elsewhere). The reference's f32 XLA matmuls default to a
# single bf16 pass on TPU, so evaluate the reference at highest precision (true math).
INTERPRET = jax.default_backend() != "tpu"


@contextlib.contextmanager
def _ref_precision():
    """The XLA reference path (kernel flag off) at highest matmul precision (the Pallas
    kernels are unaffected: they set their own precision)."""
    prev = _glm5_next._MHC_KERNEL
    _glm5_next._MHC_KERNEL = False
    try:
        with jax.default_matmul_precision("highest"):
            yield
    finally:
        _glm5_next._MHC_KERNEL = prev


def _make(t, d, seed, f32=False):
    hc = HC
    n = (2 + hc) * hc
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    dt = jnp.float32 if f32 else jnp.bfloat16
    streams = (jax.random.normal(ks[0], (t, hc * d)) * 1.5).astype(dt)
    y = jax.random.normal(ks[1], (t, d)).astype(dt)
    fn = (jax.random.normal(ks[2], (hc * d, n)) * (2.0 / np.sqrt(hc * d))).astype(dt)
    base = jax.random.normal(ks[3], (n,)).astype(jnp.float32)
    scale = jax.random.uniform(ks[4], (3,), minval=0.5, maxval=2.0).astype(jnp.float32)
    mesh = jax.make_mesh((1, 1), ("data", "tensor"), devices=jax.devices()[:1])
    hcm = Glm5NextHC(d, hc, ITERS, HC_EPS, RMS_EPS, mesh, dt)
    # Load through the weight-loading recipe (checkpoint layout: fn is [n, hc*D]).
    outputs = hcm._prepare_weights([np.asarray(fn.T), np.asarray(base), np.asarray(scale)])
    params = (hcm.fn, hcm.base, hcm.scale, hcm.fn_t, hcm.scale_rows, hcm.base_rows)
    for param, value in zip(params, outputs):
        param[...] = value
    return hcm, streams, y


def _unsharded(hcm):
    """fn, scale, base as plain arrays for direct kernel calls (the params carry mesh shardings)."""
    return tuple(jnp.asarray(np.asarray(p[...])) for p in (hcm.fn, hcm.scale, hcm.base))


def test_load_recipe_matches_prepare_helpers():
    """The load-time recipe builds the same kernel layout as prepare_fn_t / prepare_params."""
    from sgl_jax.srt.kernels.mhc.glm5_mhc import prepare_fn_t, prepare_params

    hcm, _, _ = _make(8, 256, seed=3)
    np.testing.assert_array_equal(
        np.asarray(hcm.fn_t.value), np.asarray(prepare_fn_t(hcm.fn.value, HC))
    )
    sc, bs = prepare_params(hcm.scale.value, hcm.base.value, HC)
    np.testing.assert_array_equal(np.asarray(hcm.scale_rows.value), np.asarray(sc))
    np.testing.assert_array_equal(np.asarray(hcm.base_rows.value), np.asarray(bs))


def _np_sinkhorn(logits, iters, eps):
    """Independent float64 reference of the exact variant used by glm53 (eps placement)."""
    c = np.exp(logits - logits.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + eps
    c = c / (c.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdims=True) + eps)
        c = c / (c.sum(-2, keepdims=True) + eps)
    return c


@pytest.mark.parametrize("t,d", [(4, 4096), (16, 4096), (64, 4096), (100, 4096), (512, 1024)])
def test_mhc_kernel_matches_reference(t, d):
    hcm, streams, y = _make(t, d, seed=t)
    with _ref_precision():
        collapsed_ref, (post_ref, comb_ref) = hcm.pre(streams)
        out_ref = hcm.post(y, (post_ref, comb_ref), streams)

    collapsed, state = glm5_mhc_pre(
        streams,
        *_unsharded(hcm),
        hc=HC,
        hc_iters=ITERS,
        hc_eps=HC_EPS,
        eps=hcm.eps,
        interpret=INTERPRET,
    )
    pre, post, comb = unpack_state(state, HC)
    out = glm5_mhc_post(y, state, streams, hc=HC, interpret=INTERPRET)

    assert collapsed.shape == (t, d) and collapsed.dtype == streams.dtype
    assert out.shape == streams.shape and out.dtype == streams.dtype
    np.testing.assert_allclose(np.asarray(post), np.asarray(post_ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(comb), np.asarray(comb_ref), rtol=1e-4, atol=1e-5)
    # bf16 outputs: allow ~1 bf16 ulp from differing f32 summation order before rounding.
    f = lambda a: np.asarray(a.astype(jnp.float32))
    np.testing.assert_allclose(f(collapsed), f(collapsed_ref), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(f(out), f(out_ref), rtol=1e-2, atol=2e-2)

    # pre gate: recompute from the reference mix and compare with the kernel's.
    flat = np.asarray(streams.astype(jnp.float32), np.float64)
    flat = flat / np.sqrt((flat * flat).mean(-1, keepdims=True) + hcm.eps)
    mix = flat @ np.asarray(hcm.fn[...].astype(jnp.float32), np.float64)
    sc, bs = np.asarray(hcm.scale[...], np.float64), np.asarray(hcm.base[...], np.float64)
    pre64 = 1 / (1 + np.exp(-(mix[:, :HC] * sc[0] + bs[:HC]))) + HC_EPS
    np.testing.assert_allclose(np.asarray(pre), pre64, rtol=1e-4, atol=1e-5)
    # independent float64 Sinkhorn (exact eps placement) on the comb matrix
    logits = mix[:, 2 * HC :].reshape(t, HC, HC) * sc[2] + bs[2 * HC :].reshape(HC, HC)
    np.testing.assert_allclose(
        np.asarray(comb), _np_sinkhorn(logits, ITERS, HC_EPS), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("bt", [16, 32])
def test_mhc_kernel_block_sizes_partial_tail(bt):
    t, d = 72, 512  # 72 % 32 != 0 -> partial edge block
    hcm, streams, y = _make(t, d, seed=7)
    with _ref_precision():
        collapsed_ref, st_ref = hcm.pre(streams)
        out_ref = hcm.post(y, st_ref, streams)
    collapsed, state = glm5_mhc_pre(
        streams,
        *_unsharded(hcm),
        hc=HC,
        hc_iters=ITERS,
        hc_eps=HC_EPS,
        eps=hcm.eps,
        block_tokens=bt,
        interpret=INTERPRET,
    )
    out = glm5_mhc_post(y, state, streams, hc=HC, block_tokens=bt, interpret=INTERPRET)
    _, post, comb = unpack_state(state, HC)
    assert np.isfinite(np.asarray(state)).all()
    np.testing.assert_allclose(np.asarray(post), np.asarray(st_ref[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(comb), np.asarray(st_ref[1]), rtol=1e-4, atol=1e-5)
    f = lambda a: np.asarray(a.astype(jnp.float32))
    np.testing.assert_allclose(f(collapsed), f(collapsed_ref), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(f(out), f(out_ref), rtol=1e-2, atol=2e-2)


def test_mhc_kernel_f32_inputs():
    t, d = 24, 256
    hcm, streams, y = _make(t, d, seed=3, f32=True)
    with _ref_precision():
        collapsed_ref, st_ref = hcm.pre(streams)
        out_ref = hcm.post(y, st_ref, streams)
    collapsed, state = glm5_mhc_pre(
        streams,
        *_unsharded(hcm),
        hc=HC,
        hc_iters=ITERS,
        hc_eps=HC_EPS,
        eps=hcm.eps,
        block_tokens=16,
        interpret=INTERPRET,
    )
    out = glm5_mhc_post(y, state, streams, hc=HC, block_tokens=16, interpret=INTERPRET)
    np.testing.assert_allclose(
        np.asarray(collapsed), np.asarray(collapsed_ref), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(out_ref), rtol=1e-4, atol=1e-4)


def test_glm5_next_hc_flag_path(monkeypatch):
    """Glm5NextHC.pre/post with SGLANG_JAX_MHC_KERNEL on, under a ('data', 'tensor') mesh + jit."""
    from sgl_jax.srt.models import glm5_next

    t, d = 32, 512
    hcm, streams, y = _make(t, d, seed=11)
    with _ref_precision():
        ref_c, ref_st = hcm.pre(streams)
        ref_out = hcm.post(y, ref_st, streams)

    n_dev = len(jax.devices())
    dp = 2 if n_dev >= 4 else 1
    tp = 2 if n_dev >= 4 else 1
    mesh = jax.make_mesh((dp, tp), ("data", "tensor"), devices=jax.devices()[: dp * tp])
    monkeypatch.setattr(glm5_next, "_MHC_KERNEL", True)
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
