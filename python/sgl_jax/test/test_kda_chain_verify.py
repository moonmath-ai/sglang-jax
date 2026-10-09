"""Fused KDA verify kernel vs an XLA reference (per-step short conv, gate, L2 norm, recurrence)."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda.chain_verify import kda_chain_verify
from sgl_jax.srt.kernels.kda.naive import naive_recurrent_kda
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

INTERPRET = jax.default_backend() != "tpu"
B, T, H, K, SLOTS, W = 3, 4, 2, 128, 7, 4
LOWER_BOUND = -5.0


def _l2(x):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.sum(x32 * x32, -1, keepdims=True) + 1e-6)).astype(x.dtype)


def _gate(a, a_log, dt_bias):
    g32 = a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    a_scale = jnp.exp(a_log.reshape(H, 1).astype(jnp.float32))
    return (LOWER_BOUND * jax.nn.sigmoid(a_scale * g32)).astype(a.dtype)


def _case(seed):
    rng = np.random.default_rng(seed)
    bf = jnp.bfloat16
    x = [jnp.asarray(rng.normal(size=(B, T, H, K)), bf) for _ in range(3)]  # pre-conv q, k, v
    return dict(
        x=x,
        a=jnp.asarray(rng.normal(size=(B, T, H, K)), bf),
        beta=jnp.asarray(rng.uniform(0.1, 0.9, size=(B, T, H)), jnp.float32),
        a_log=jnp.asarray(rng.normal(size=(H,)) * 0.3, jnp.float32),
        dt_bias=jnp.asarray(rng.normal(size=(H, K)) * 0.3, jnp.float32),
        ssm_pool=jnp.asarray(rng.normal(size=(SLOTS, H, K, K)) * 0.1, jnp.float32),
        conv_pool=jnp.asarray(rng.normal(size=(SLOTS, W - 1, 3, H, K)), bf),
        conv_w=jnp.asarray(rng.normal(size=(W, 3, H, K)) * 0.5, bf),
        st=jnp.asarray(rng.normal(size=(SLOTS, T, H, K, K)), jnp.float32),
        stc=jnp.asarray(rng.normal(size=(SLOTS, T, W - 1, 3, H, K)), bf),
    )


def _reference(c, h0, hist):
    """h0 [B, H, K, K] / hist [B, W-1, 3, H, K] start states -> (o, step states, step windows)."""
    D = H * K
    posts, windows = [], []
    for i, x in enumerate(c["x"]):
        cache = jnp.transpose(hist[:, :, i].reshape(B, W - 1, D), (0, 2, 1))  # [B, D, W-1]
        w = jnp.transpose(c["conv_w"][:, i].reshape(W, D))  # [D, W]
        ys, cs = [], []
        for t in range(T):
            y, cache = short_convolution(
                x[:, t].reshape(B, D), w, cache, None, ForwardMode.DECODE, activation="silu"
            )
            ys.append(y.reshape(B, H, K))
            cs.append(jnp.transpose(cache, (0, 2, 1)).reshape(B, W - 1, H, K))
        posts.append(jnp.stack(ys, 1))
        windows.append(jnp.stack(cs, 1))
    q, k, v = _l2(posts[0]), _l2(posts[1]), posts[2]
    g = _gate(c["a"], c["a_log"], c["dt_bias"])
    h, outs, states = h0, [], []
    for t in range(T):
        o, h = naive_recurrent_kda(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            g[:, t : t + 1],
            c["beta"][:, t : t + 1],
            scale=K**-0.5,
            initial_state=h,
            output_final_state=True,
        )
        outs.append(o[:, 0])
        states.append(h)
    return jnp.stack(outs, 1), jnp.stack(states, 1), jnp.stack(windows, axis=3)


def _check(o, st_rows, stc_rows, o_ref, st_ref, stc_ref):
    np.testing.assert_allclose(
        np.asarray(o, np.float32), np.asarray(o_ref, np.float32), rtol=3e-2, atol=3e-2
    )
    np.testing.assert_allclose(np.asarray(st_rows), np.asarray(st_ref), rtol=2e-2, atol=2e-2)
    np.testing.assert_array_equal(np.asarray(stc_rows, np.float32), np.asarray(stc_ref, np.float32))


def _run(c, idx):
    return kda_chain_verify(
        *c["x"],
        c["a"],
        c["beta"],
        c["a_log"],
        c["dt_bias"],
        c["conv_w"],
        c["conv_pool"],
        c["ssm_pool"],
        idx,
        c["st"],
        c["stc"],
        scale=K**-0.5,
        lower_bound=LOWER_BOUND,
        interpret=INTERPRET,
    )


def test_batch_rows_scratch_matches_reference():
    c = _case(5)
    idx = jnp.asarray([4, 0, 2], jnp.int32)  # slot 0 = padding row
    o, st, stc = _run(c, idx)
    _check(o, st[:B], stc[:B], *_reference(c, c["ssm_pool"][idx], c["conv_pool"][idx]))
    np.testing.assert_array_equal(np.asarray(st[B:]), np.asarray(c["st"][B:]))  # untouched
