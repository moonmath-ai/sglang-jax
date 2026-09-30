"""Fused KDA chain-verify kernel vs the per-step reference recurrence."""

import os

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.kda.chain_verify import kda_chain_verify
from sgl_jax.srt.kernels.kda.naive import naive_recurrent_kda

INTERPRET = jax.default_backend() != "tpu"


def _l2(x):
    return x / jnp.sqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)


def _case(B=3, T=4, H=2, K=128, V=128, R=5, seed=0, dtype=jnp.bfloat16):
    rng = np.random.default_rng(seed)
    q = _l2(jnp.asarray(rng.normal(size=(B, T, H, K)), jnp.float32)).astype(dtype)
    k = _l2(jnp.asarray(rng.normal(size=(B, T, H, K)), jnp.float32)).astype(dtype)
    v = jnp.asarray(rng.normal(size=(B, T, H, V)), dtype)
    g = jnp.asarray(-np.abs(rng.normal(size=(B, T, H, K))) * 0.3, dtype)
    beta = jnp.asarray(rng.uniform(0.1, 0.9, size=(B, T, H)), dtype)
    h0 = jnp.asarray(rng.normal(size=(B, H, K, V)) * 0.1, jnp.float32)
    scratch = jnp.asarray(rng.normal(size=(R, T, H, K, V)), jnp.float32)
    return q, k, v, g, beta, h0, scratch


def _reference(q, k, v, g, beta, h0, scale):
    B, T = q.shape[:2]
    h, outs, states = h0, [], []
    for t in range(T):
        o, h = naive_recurrent_kda(
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1], g[:, t : t + 1],
            beta[:, t : t + 1], scale=scale, initial_state=h, output_final_state=True,
        )
        outs.append(o[:, 0])
        states.append(h)
    return jnp.stack(outs, 1), jnp.stack(states, 1)  # [B,T,H,V], [B,T,H,K,V]


def _to_bhtd(x):
    return jnp.transpose(x, (0, 2, 1, 3))


def _to_bhkt(x):
    return jnp.transpose(x, (0, 2, 3, 1))


def test_chain_verify_matches_reference():
    q, k, v, g, beta, h0, scratch = _case()
    B, T, H, K = q.shape
    scale = K**-0.5
    o_ref, st_ref = _reference(q, k, v, g, beta, h0, scale)
    kb = (k.astype(jnp.float32) * beta.astype(jnp.float32)[..., None]).astype(k.dtype)
    qs = (q.astype(jnp.float32) * scale).astype(q.dtype)
    o, st = kda_chain_verify(
        _to_bhkt(qs), _to_bhkt(k), _to_bhkt(kb), _to_bhtd(v), _to_bhkt(g), h0, scratch,
        interpret=INTERPRET,
    )
    o = jnp.transpose(o, (0, 2, 1, 3))
    np.testing.assert_allclose(np.asarray(o, np.float32), np.asarray(o_ref, np.float32),
                               rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(np.asarray(st[:B]), np.asarray(st_ref), rtol=2e-2, atol=2e-2)
    # rows beyond the batch are untouched
    np.testing.assert_array_equal(np.asarray(st[B:]), np.asarray(scratch[B:]))


if __name__ == "__main__":
    test_chain_verify_matches_reference()
    print("ok")
