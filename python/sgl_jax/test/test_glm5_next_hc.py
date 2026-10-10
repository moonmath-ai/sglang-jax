"""GLM-5.3-Flash mHC (``Glm5NextHC`` on the DeepSeek-V4 mHC kernels) vs an f32 reference of the
GLM math, and the v7x tile / backend selection.

JAX_PLATFORMS=cpu PALLAS_INTERPRET=1 JAX_NUM_CPU_DEVICES=4 \
    python -m pytest -q python/sgl_jax/test/test_glm5_next_hc.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.mhc import tune
from sgl_jax.srt.models.glm5_next import Glm5NextHC

HC, ITERS, HC_EPS, RMS_EPS = 4, 20, 1e-6, 1e-5


@pytest.fixture(autouse=True)
def _interpret_off_tpu(monkeypatch):
    if jax.default_backend() != "tpu":
        monkeypatch.setenv("PALLAS_INTERPRET", "1")
        # The DSV4 kernels pick tiles per TPU generation; on CPU use the v6e table.
        monkeypatch.setattr(
            "sgl_jax.srt.kernels.mhc.mhc._device_kind", lambda: "TPU v6 lite", raising=True
        )


def _ref(streams, y, fn, scale, base):
    """f32 reference (HIGHEST precision) of the GLM-5.3 mHC math: -> (collapsed, post, comb, out).
    fn is the checkpoint's [n, hc*D]."""
    lead = streams.shape[:-1]
    d = streams.shape[-1] // HC
    flat = streams.astype(jnp.float32)
    flat = flat * jax.lax.rsqrt(jnp.mean(flat * flat, -1, keepdims=True) + RMS_EPS)
    mix = jnp.dot(flat, fn.astype(jnp.float32).T, precision="highest")
    pre_w, post_w, comb_w = mix[..., :HC], mix[..., HC : 2 * HC], mix[..., 2 * HC :]
    pre = jax.nn.sigmoid(pre_w * scale[0] + base[:HC]) + HC_EPS
    post = 2.0 * jax.nn.sigmoid(post_w * scale[1] + base[HC : 2 * HC])
    logits = comb_w.reshape(*lead, HC, HC) * scale[2] + base[2 * HC :].reshape(HC, HC)
    comb = jax.nn.softmax(logits, -1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
    s = streams.reshape(*lead, HC, d).astype(jnp.float32)
    collapsed = jnp.einsum("...h,...hd->...d", pre, s, precision="highest")
    mixed = jnp.einsum("...ij,...id->...jd", comb, s, precision="highest")
    out = post[..., None] * y.astype(jnp.float32)[..., None, :] + mixed
    return collapsed, post, comb, out.reshape(*lead, HC * d)


def _make(t, d, seed, mesh):
    n = (2 + HC) * HC
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    streams = (jax.random.normal(ks[0], (t, HC * d)) * 1.5).astype(jnp.bfloat16)
    y = jax.random.normal(ks[1], (t, d)).astype(jnp.bfloat16)
    fn = (jax.random.normal(ks[2], (n, HC * d)) * (2.0 / np.sqrt(HC * d))).astype(jnp.bfloat16)
    base = jax.random.normal(ks[3], (n,)).astype(jnp.float32)
    scale = jax.random.uniform(ks[4], (3,), minval=0.5, maxval=2.0).astype(jnp.float32)
    hcm = Glm5NextHC(d, HC, ITERS, HC_EPS, RMS_EPS, mesh, jnp.bfloat16)
    for param, value in ((hcm.fn, fn), (hcm.base, base), (hcm.scale, scale)):
        param[...] = jax.device_put(value, param.value.sharding)
    return hcm, streams, y, (fn, scale, base)


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.abs(a - b).max() / np.abs(b).max()


# T=100 / 300: partial tail blocks; with >= 4 devices tokens are also sharded over 'data' as
# under DP attention (with tensor parallelism replicating the features).
@pytest.mark.parametrize("t,d", [(4, 512), (16, 512), (100, 512), (300, 256)])
def test_glm5_next_hc_matches_reference(t, d):
    n_dev = len(jax.devices())
    dp = tp = 2 if n_dev >= 4 and t % 2 == 0 else 1
    mesh = jax.make_mesh((dp, tp), ("data", "tensor"), devices=jax.devices()[: dp * tp])
    hcm, streams, y, weights = _make(t, d, seed=t, mesh=mesh)
    ref_c, ref_post, ref_comb, ref_out = _ref(streams, y, *weights)
    if dp > 1:
        tok = jax.NamedSharding(mesh, jax.P("data", None))
        streams, y = jax.device_put(streams, tok), jax.device_put(y, tok)

    @jax.jit
    def run(s, yy):
        c, state = hcm.pre(s)
        return c, state, hcm.post(yy, state, s)

    c, (post, comb), out = run(streams, y)
    assert c.shape == (t, d) and c.dtype == jnp.bfloat16
    assert out.shape == (t, HC * d) and out.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(post), np.asarray(ref_post), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(comb), np.asarray(ref_comb), rtol=1e-3, atol=1e-5)
    # bf16 outputs: within a few bf16 ulps of the f32 reference.
    assert _rel(c, ref_c) < 1e-2
    assert _rel(out, ref_out) < 1e-2


def test_v7x_selection():
    kw = dict(hc_mult=HC, hidden=4096)
    # Collapse tile 128 fits v7x's 32 MiB scoped VMEM (measured best at this shape).
    assert tune.select_collapse_block_tokens("TPU7x", tokens=4096, activation_bytes=2, **kw) == 128
    # XLA post at every size on v7x; v6e switches to Pallas once XLA would spill.
    for tokens in (16, 512, 4096, 16384):
        assert tune.select_post_backend("TPU7x", tokens=tokens, activation_bytes=2, **kw) == "xla"
    assert (
        tune.select_post_backend("TPU v6 lite", tokens=4096, activation_bytes=2, **kw) == "pallas"
    )
