"""Fused KDA layer core for speculative target verify (topk=1 chains).

One Pallas program per request runs, for all local heads and the request's ``T`` chain
tokens: the depthwise causal short conv (+ SiLU) over [conv history | chain tokens], the
gate (``dt_bias`` / ``A_log``), q/k L2 norm and scale, and the delta-rule recurrence with the
state resident in VMEM. The state and the conv window after every step are written straight
into the verify scratch (aliased in place) so the commit can keep the accepted step.

Per step (same math as ``naive_recurrent_kda``, fp32 state):
    S = S * exp(g_t)[:, None]
    S = S + (beta_t * k_t)[:, None] * (v_t - k_t @ S)
    o_t = (q_t * scale) @ S

Inputs keep the model's [T, H, D] layout; prep runs per step on whole [H, D] tiles and one
transpose per tensor per step yields [D, H], whose column h is head h's vector (contractions
are sublane reductions, no 1-row matmuls). The conv history is stored taps-major
([K-1, 3, H, D], one [H, D] tile per tap and q/k/v), so it is read without relayouts.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _verify_body(
    ascale_ref,
    q_ref,
    k_ref,
    v_ref,
    a_ref,
    beta_ref,
    dtb_ref,
    convw_ref,
    hist_tap,
    h0_head,
    o_ref,
    st_ref,
    stc_ref,
    *,
    scale,
    lower_bound,
    x_dtype,
    g_dtype,
):
    T, H = q_ref.shape[1], q_ref.shape[2]
    km1 = stc_ref.shape[2]
    f32 = jnp.float32
    ascale = ascale_ref[...]  # [H, 1] exp(A_log)
    dtb = dtb_ref[...].astype(f32)  # [H, D]

    def conv_steps(c, x_ref):
        """Post-conv [H, D] tiles of tensor c for every step (rounded like the XLA conv);
        writes the post-step window (last K-1 taps) of every step to the conv scratch."""
        win = [hist_tap(j, c) for j in range(km1)] + [x_ref[0, t] for t in range(T)]
        w = [convw_ref[j, c].astype(f32) for j in range(km1 + 1)]
        ys = []
        for t in range(T):
            acc = sum(win[t + j].astype(f32) * w[j] for j in range(km1 + 1))
            acc = acc.astype(x_dtype).astype(f32)
            ys.append(jax.nn.silu(acc).astype(x_dtype).astype(f32))
            for j in range(km1):
                stc_ref[0, t, j, c] = win[t + 1 + j].astype(stc_ref.dtype)
        return ys

    def l2(x):
        x = x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)
        return x.astype(x_dtype).astype(f32)

    qs, ks, vs = conv_steps(0, q_ref), conv_steps(1, k_ref), conv_steps(2, v_ref)
    qT, kT, kbT, dT = [], [], [], []
    for t in range(T):
        g = a_ref[0, t].astype(f32) + dtb
        if lower_bound is None:
            g = -ascale * jax.nn.softplus(g)
        else:
            g = lower_bound * jax.nn.sigmoid(ascale * g)
        g = g.astype(g_dtype).astype(f32)
        kt = jnp.transpose(l2(ks[t]))  # [D, H]
        qT.append(jnp.transpose(l2(qs[t]) * scale))
        kT.append(kt)
        kbT.append(kt * beta_ref[0, t : t + 1, :].astype(f32))  # beta row [1, H]
        dT.append(jnp.transpose(jnp.exp(g)))
    outs = [[None] * H for _ in range(T)]
    for h in range(H):
        S = h0_head(h).astype(f32)  # [D, V]
        for t in range(T):
            S = S * dT[t][:, h : h + 1]
            pred = jnp.sum(kT[t][:, h : h + 1] * S, axis=0, keepdims=True)  # [1, V]
            S = S + kbT[t][:, h : h + 1] * (vs[t][h : h + 1, :] - pred)
            outs[t][h] = jnp.sum(qT[t][:, h : h + 1] * S, axis=0, keepdims=True)
            st_ref[0, t, h] = S.astype(st_ref.dtype)
    for t in range(T):
        o_ref[0, t] = jnp.concatenate(outs[t], axis=0).astype(o_ref.dtype)


def _batch_rows_kernel(
    idx_ref,
    ascale_ref,
    q_ref,
    k_ref,
    v_ref,
    a_ref,
    beta_ref,
    dtb_ref,
    convw_ref,
    hist_ref,
    h0_ref,
    st_in_ref,
    stc_in_ref,
    o_ref,
    st_ref,
    stc_ref,
    **kw,
):
    """Running state in the pool; scratch rows are batch rows."""
    del idx_ref, st_in_ref, stc_in_ref  # idx drives the pool index_maps; scratch is aliased
    _verify_body(
        ascale_ref,
        q_ref,
        k_ref,
        v_ref,
        a_ref,
        beta_ref,
        dtb_ref,
        convw_ref,
        lambda j, c: hist_ref[0, j, c],
        lambda h: h0_ref[0, h],
        o_ref,
        st_ref,
        stc_ref,
        **kw,
    )


def _slot_rows_kernel(
    idx_ref,
    step_ref,
    ascale_ref,
    q_ref,
    k_ref,
    v_ref,
    a_ref,
    beta_ref,
    dtb_ref,
    convw_ref,
    hist_p,
    h0_p,
    h0_s,
    hist_s,
    o_ref,
    st_ref,
    stc_ref,
    **kw,
):
    """Lazy commit: scratch rows are slots; a row whose step >= 0 holds its running state in
    scratch step ``step`` of its slot (seed from there), else in the pool."""
    use_s = step_ref[pl.program_id(0)] >= 0
    _verify_body(
        ascale_ref,
        q_ref,
        k_ref,
        v_ref,
        a_ref,
        beta_ref,
        dtb_ref,
        convw_ref,
        lambda j, c: jnp.where(use_s, hist_s[0, 0, j, c], hist_p[0, j, c]),
        lambda h: jnp.where(use_s, h0_s[0, 0, h], h0_p[0, h]),
        o_ref,
        st_ref,
        stc_ref,
        **kw,
    )


@functools.partial(jax.jit, static_argnames=("scale", "lower_bound", "g_dtype", "interpret"))
def kda_chain_verify(
    q,
    k,
    v,
    a,
    beta,
    a_log,
    dt_bias,
    conv_w,
    conv_pool,
    ssm_pool,
    slot_idx,
    step_states,
    step_conv,
    spec_step=None,
    *,
    scale,
    lower_bound=None,
    g_dtype=None,
    interpret=None,
):
    """Verify core on raw inputs (local shard).

    Args:
      q, k, v: [B, T, H, D] **pre-conv** projections. a: [B, T, H, D] raw gate.
      beta: [B, T, H]. a_log: [H]. dt_bias: [H, D].
      conv_w: [W, 3, H, D] depthwise taps for q/k/v. conv_pool: [slots, W-1, 3, H, D] running
        conv history (row ``slot_idx[b]`` is request b's). ssm_pool: [slots, H, D, D].
      step_states: [R, T, H, D, D] / step_conv: [R, T, W-1, 3, H, D] verify scratch.
      spec_step: optional [slots] int32 (lazy commit). None: scratch rows are batch rows
        0..B-1 and the running state is the pool row. Given: scratch rows are **slots**, and a
        slot with ``spec_step >= 0`` seeds from scratch step ``spec_step`` instead of the pool.

    Returns (o [B, T, H, D], updated step_states, updated step_conv).
    """
    if interpret is None:  # Pallas TPU kernels run in interpret mode off-TPU (CPU tests)
        interpret = jax.default_backend() != "tpu"
    B, T, H, K = q.shape
    V = v.shape[-1]
    km1 = conv_pool.shape[1]
    lazy = spec_step is not None
    assert conv_pool.shape[2:] == (3, H, K) and conv_w.shape == (km1 + 1, 3, H, K)
    assert step_states.shape[1:] == (T, H, K, V) and step_conv.shape[1:] == (T, km1, 3, H, K)
    if not lazy:
        assert step_states.shape[0] >= B and step_conv.shape[0] >= B
    kw = dict(
        scale=float(scale),
        lower_bound=None if lower_bound is None else float(lower_bound),
        x_dtype=q.dtype,
        g_dtype=jnp.dtype(g_dtype or a.dtype),
    )
    # index_maps get (grid index, *scalar-prefetch refs); slot_idx is always the first ref.
    tok = lambda b, *_: (b, 0, 0, 0)
    const = lambda nd: (lambda b, *_: (0,) * nd)
    slot = lambda nd: (lambda b, idx, *_: (idx[b],) + (0,) * (nd - 1))
    in_specs = [
        pl.BlockSpec((H, 1), const(2)),  # exp(A_log)
        pl.BlockSpec((1, T, H, K), tok),
        pl.BlockSpec((1, T, H, K), tok),
        pl.BlockSpec((1, T, H, V), tok),
        pl.BlockSpec((1, T, H, K), tok),
        pl.BlockSpec((1, T, H), lambda b, *_: (b, 0, 0)),
        pl.BlockSpec((H, K), const(2)),
        pl.BlockSpec((km1 + 1, 3, H, K), const(4)),
        pl.BlockSpec((1, km1, 3, H, K), slot(5)),  # pool conv history
        pl.BlockSpec((1, H, K, V), slot(4)),  # pool state
    ]
    if lazy:
        at_step = lambda nd: (lambda b, idx, st: (idx[b], jnp.maximum(st[b], 0)) + (0,) * (nd - 2))
        in_specs += [
            pl.BlockSpec((1, 1, H, K, V), at_step(5)),  # scratch state (aliased)
            pl.BlockSpec((1, 1, km1, 3, H, K), at_step(6)),  # scratch conv window (aliased)
        ]
        rows = slot
        kernel = _slot_rows_kernel
        # Per-row seed step (gathered here: indexing the whole per-slot table by slot inside
        # the index_map halted the TPU core).
        prefetch = (slot_idx.astype(jnp.int32), spec_step[slot_idx].astype(jnp.int32))
    else:
        in_specs += [pl.BlockSpec(memory_space=pl.ANY), pl.BlockSpec(memory_space=pl.ANY)]
        rows = lambda nd: (lambda b, *_: (b,) + (0,) * (nd - 1))
        kernel = _batch_rows_kernel
        prefetch = (slot_idx.astype(jnp.int32),)
    n_pre = len(prefetch)
    return pl.pallas_call(
        functools.partial(kernel, **kw),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=n_pre,
            grid=(B,),
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec((1, T, H, V), tok),
                pl.BlockSpec((1, T, H, K, V), rows(5)),
                pl.BlockSpec((1, T, km1, 3, H, K), rows(6)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((B, T, H, V), v.dtype),
            jax.ShapeDtypeStruct(step_states.shape, step_states.dtype),
            jax.ShapeDtypeStruct(step_conv.shape, step_conv.dtype),
        ],
        # operands: *prefetch, ascale, q, k, v, a, beta, dtb, conv_w, conv_pool, ssm_pool, st, stc
        input_output_aliases={n_pre + 10: 1, n_pre + 11: 2},
        compiler_params=pltpu.CompilerParams(
            # lazy: a slot's scratch is read (seed) and written in the same program
            dimension_semantics=("arbitrary",) if lazy else ("parallel",)
        ),
        interpret=interpret,
    )(
        *prefetch,
        jnp.exp(a_log.astype(jnp.float32)).reshape(H, 1),
        q,
        k,
        v,
        a,
        beta.astype(jnp.float32),
        dt_bias.astype(jnp.float32),
        conv_w,
        conv_pool,
        ssm_pool,
        step_states,
        step_conv,
    )
