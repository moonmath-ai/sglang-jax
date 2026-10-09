"""Grouped-matmul EP-MoE for large (prefill) token counts, on FusedEPMoE weights.

The fused v2 kernel is tuned for decode-sized batches: at 4096 tokens it spends ~4.2 ms per
GLM-5.3-Flash layer, while each device's expert FFN is ~0.2 TFLOP (~0.65 ms in megablox gmm).
This path keeps tokens replicated and experts sharded over the whole mesh (the fused kernel's
weight layout): every device sorts the routed slots by expert, gathers only the rows of its own
experts (SparseCore ragged gather), runs the three expert matmuls with ``gmm`` (group_offset
selects the local experts), combines its rows back into token order, adds its slice of the
shared expert, and a psum over the mesh sums the partial outputs.

Numerics follow ``fused_moe.v2.kernel.ref_moe``: ``activation`` gate clamped at ``swiglu_limit``
(single sided), up branch clamped to +-limit; FP8 activations quantized per token when ``act_quant``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v2.kernel import activation_fn
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.kernels.sparse_core.moe_permute import sc_combine, sc_dispatch_gather

_AXES = ("data", "tensor")


def _shared_expert(x, w1, w3, w2, s1, s3, s2, activation, limit, out_dtype):
    def deq(w, s):
        w = w.astype(jnp.bfloat16)
        return w if s is None else (w.astype(jnp.float32) * s.reshape(1, -1)).astype(jnp.bfloat16)

    f32 = jnp.float32
    gate = jnp.dot(x, deq(w1, s1), preferred_element_type=f32)
    up = jnp.dot(x, deq(w3, s3), preferred_element_type=f32)
    act = activation_fn(gate, up, activation, limit).astype(jnp.bfloat16)
    return jnp.dot(act, deq(w2, s2), preferred_element_type=f32).astype(out_dtype)


def gmm_prefill_moe(
    mesh,
    hidden_states,
    topk_weights,
    topk_ids,
    w1,
    w3,
    w2,
    w1_scale,
    w3_scale,
    w2_scale,
    w1_shared=None,
    w3_shared=None,
    w2_shared=None,
    w1_shared_scale=None,
    w3_shared_scale=None,
    w2_shared_scale=None,
    *,
    top_k: int,
    activation: str = "silu",
    swiglu_limit: float | None,
    shared_swiglu_limit: float | None,
    act_quant: bool,
    scatter_tokens: bool = False,
):
    """``hidden_states`` [T, H] (any sharding); returns [T, H] replicated over the mesh, or
    token-sharded over ('data', 'tensor') with ``scatter_tokens`` (reduce-scatter instead of
    all-reduce, for sequence-parallel callers; needs T divisible by the mesh size).

    ``w1/w3`` [E, H, I], ``w2`` [E, I, H] expert-sharded over ('data', 'tensor'); block scales
    [E, K // qbk, 1, N]; shared expert [H, Is] / [Is, H] with per-channel scales (1, 1, Is) /
    (1, 1, H), or None.
    """
    n_shards = mesh.shape["data"] * mesh.shape["tensor"]
    E = w1.shape[0]
    assert E % n_shards == 0, (E, n_shards)
    e_loc = E // n_shards
    T = hidden_states.shape[0]
    out_dtype = hidden_states.dtype
    has_shared = w1_shared is not None
    split_shared = has_shared and T % n_shards == 0

    def _local(x, tw, ti, w1_, w3_, w2_, s1, s3, s2, w1s, w3s, w2s, s1s, s3s, s2s):
        d = jax.lax.axis_index(_AXES)
        flat = ti.reshape(-1).astype(jnp.int32)
        order = jnp.argsort(flat, stable=True)
        group_sizes = jnp.bincount(flat, length=E).astype(jnp.int32)
        off = (d * e_loc).astype(jnp.int32)
        csum = jnp.cumsum(group_sizes)
        start = jnp.where(off == 0, 0, csum[jnp.maximum(off - 1, 0)]).astype(jnp.int32)
        end = csum[off + e_loc - 1].astype(jnp.int32)
        xs = sc_dispatch_gather(x, order // top_k, start, end).astype(out_dtype)

        kw = dict(
            group_sizes=group_sizes,
            preferred_element_type=out_dtype,
            group_offset=off,
            maybe_quantize_lhs=act_quant,
            acc_dtype=jnp.float32,
        )
        gate = gmm(xs, w1_, rhs_scale=s1, zero_initialize=False, **kw)
        up = gmm(xs, w3_, rhs_scale=s3, zero_initialize=False, **kw)
        act = activation_fn(
            gate.astype(jnp.float32), up.astype(jnp.float32), activation, swiglu_limit
        ).astype(out_dtype)
        y = gmm(act, w2_, rhs_scale=s2, zero_initialize=True, **kw)  # non-local rows = 0

        inv = jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=order.dtype))
        valid = (flat >= off) & (flat < off + e_loc)
        out = sc_combine(y, inv, tw.reshape(-1).astype(jnp.float32), valid, top_k)
        out = out.astype(out_dtype)

        if has_shared:
            if split_shared:  # each device adds the shared expert for its slice of tokens
                rows = T // n_shards
                xr = jax.lax.dynamic_slice_in_dim(x, d * rows, rows, axis=0)
                se = _shared_expert(
                    xr, w1s, w3s, w2s, s1s, s3s, s2s, activation, shared_swiglu_limit, out_dtype
                )
                cur = jax.lax.dynamic_slice_in_dim(out, d * rows, rows, axis=0)
                out = jax.lax.dynamic_update_slice_in_dim(out, cur + se, d * rows, axis=0)
            else:
                se = _shared_expert(
                    x, w1s, w3s, w2s, s1s, s3s, s2s, activation, shared_swiglu_limit, out_dtype
                )
                out = out + jnp.where(d == 0, se, jnp.zeros_like(se))
        if scatter_tokens:
            return jax.lax.psum_scatter(out, _AXES, scatter_dimension=0, tiled=True)
        return jax.lax.psum(out, _AXES)

    # Token-major inputs are consumed replicated (every device routes all tokens).
    replicated = jax.sharding.NamedSharding(mesh, P())
    hidden_states, topk_weights, topk_ids = (
        jax.sharding.reshard(a, replicated) for a in (hidden_states, topk_weights, topk_ids)
    )
    args = (
        hidden_states,
        topk_weights,
        topk_ids,
        w1,
        w3,
        w2,
        w1_scale,
        w3_scale,
        w2_scale,
        w1_shared,
        w3_shared,
        w2_shared,
        w1_shared_scale,
        w3_shared_scale,
        w2_shared_scale,
    )
    rep = lambda a: P(*([None] * a.ndim))
    exp = lambda a: P(_AXES, *([None] * (a.ndim - 1)))
    specs = (
        rep(hidden_states),
        rep(topk_weights),
        rep(topk_ids),
        *(exp(a) for a in args[3:9]),
        *(None if a is None else rep(a) for a in args[9:]),
    )
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=specs,
        out_specs=P(_AXES, None) if scatter_tokens else P(None, None),
        check_vma=False,
    )(*args)
