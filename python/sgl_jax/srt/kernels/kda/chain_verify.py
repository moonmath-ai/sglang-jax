"""Fused KDA recurrence for speculative target verify (topk=1 chains).

One Pallas program per request runs the delta-rule recurrence over the request's
``T`` chain tokens for all local heads with the state resident in VMEM, and writes
the state after every step straight into the verify scratch (aliased in place), so
the commit can pick the step of the accepted length. Replaces a per-step XLA loop
that round-tripped the state through HBM each step and then stacked/copied the
per-step states.

Per step (same math as ``naive_recurrent_kda``, fp32 state):
    S = S * exp(g_t)[:, None]
    S = S + (beta_t * k_t)[:, None] * (v_t - k_t @ S)
    o_t = (q_t * scale) @ S
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HI = jax.lax.Precision.HIGHEST


def _chain_kernel(qT_ref, kT_ref, kbT_ref, gT_ref, v_ref, h0_ref, st_in_ref, o_ref, st_ref):
    """VPU-only formulation: per-step vectors arrive as columns ([K, T] blocks), so
    they broadcast across lanes against the [K, V] state and every contraction is a
    sublane reduction -- no 1-row matmuls, no lane<->sublane relayouts, exact fp32."""
    del st_in_ref  # aliased with st_ref; rows of other requests stay untouched
    H, T = v_ref.shape[1], v_ref.shape[2]
    for h in range(H):
        S = h0_ref[0, h].astype(jnp.float32)  # [K, V]
        qT = qT_ref[0, h].astype(jnp.float32)  # [K, T] (pre-scaled)
        kT = kT_ref[0, h].astype(jnp.float32)
        kbT = kbT_ref[0, h].astype(jnp.float32)  # beta * k
        decay = jnp.exp(gT_ref[0, h].astype(jnp.float32))
        v = v_ref[0, h].astype(jnp.float32)  # [T, V]
        outs = []
        for t in range(T):
            S = S * decay[:, t : t + 1]
            pred = jnp.sum(kT[:, t : t + 1] * S, axis=0, keepdims=True)  # [1, V]
            S = S + kbT[:, t : t + 1] * (v[t : t + 1] - pred)
            outs.append(jnp.sum(qT[:, t : t + 1] * S, axis=0, keepdims=True))
            st_ref[0, t, h] = S.astype(st_ref.dtype)
        o_ref[0, h] = jnp.concatenate(outs, axis=0).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("interpret",))
def kda_chain_verify(q, k, kb, v, g, h0, step_states, *, interpret: bool | None = None):
    """Local-shard fused chain recurrence.

    Args:
      q:  [B, H, K, T] queries (transposed: per-step columns), L2-normalized and
          pre-multiplied by ``scale``.
      k:  [B, H, K, T] keys (L2-normalized).
      kb: [B, H, K, T] ``beta * k``.
      g:  [B, H, K, T] log-space decay.
      v:  [B, H, T, V] values.
      h0: [B, H, K, V] initial state (read-only).
      step_states: [R >= B, T, H, K, V] verify scratch; rows ``0..B-1`` receive the
        state after each step, other rows are preserved (in place).

    Returns:
      (o [B, H, T, V] in ``v.dtype``, updated step_states)
    """
    if interpret is None:  # Pallas TPU kernels run in interpret mode off-TPU (CPU tests)
        interpret = jax.default_backend() != "tpu"
    B, H, K, T = q.shape
    V = v.shape[-1]
    assert step_states.shape[1:] == (T, H, K, V), (step_states.shape, (T, H, K, V))
    assert step_states.shape[0] >= B
    seq = lambda b: (b, 0, 0, 0)
    in_specs = [
        pl.BlockSpec((1, H, K, T), seq),
        pl.BlockSpec((1, H, K, T), seq),
        pl.BlockSpec((1, H, K, T), seq),
        pl.BlockSpec((1, H, K, T), seq),
        pl.BlockSpec((1, H, T, V), seq),
        pl.BlockSpec((1, H, K, V), seq),
        pl.BlockSpec(memory_space=pl.ANY),  # aliased scratch, never read
    ]
    out_specs = [
        pl.BlockSpec((1, H, T, V), seq),
        pl.BlockSpec((1, T, H, K, V), lambda b: (b, 0, 0, 0, 0)),
    ]
    return pl.pallas_call(
        _chain_kernel,
        grid=(B,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=[
            jax.ShapeDtypeStruct((B, H, T, V), v.dtype),
            jax.ShapeDtypeStruct(step_states.shape, step_states.dtype),
        ],
        input_output_aliases={6: 1},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(q, k, kb, g, v, h0, step_states)
