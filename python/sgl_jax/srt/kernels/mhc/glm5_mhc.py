"""Fused Pallas TPU kernels for GLM-5.3-Flash's four-stream hyper-connection (mHC).

Replaces the per-site chain of tiny XLA ops in ``Glm5NextHC`` (models/glm5_next.py) with two
kernels, one per call site:

  * ``glm5_mhc_pre``  : RMS-normalise the [T, hc*D] streams, project with ``fn`` on the MXU,
                        compute the pre/post gates and the Sinkhorn-normalised hc x hc ``comb``
                        matrix, and collapse the streams to [T, D] -- all in one pass over the
                        streams. Returns ``(collapsed [T, D], state [T, n_pad] f32)``.
  * ``glm5_mhc_post`` : ``out[:, j] = post[j] * y + sum_i comb[i, j] * streams[:, i]``.

Math is exactly ``Glm5NextHC._mix`` / ``post`` (f32 everywhere):

    pre  = sigmoid(pre_w * scale0 + base[:hc]) + hc_eps
    post = 2 * sigmoid(post_w * scale1 + base[hc:2hc])
    comb = softmax(comb_w * scale2 + base[2hc:], axis=-1) + hc_eps
    comb = comb / (colsum + hc_eps);  (hc_iters - 1) x [row-normalise, column-normalise]

Kernel design (pre):
  * Tokens are tiled ``bt`` rows per program; the whole hc*D feature row stays in VMEM.
  * ``fn`` is re-laid-out once per call (XLA side) as ``fn_t [n_pad, hc*D]``: every gate group
    (pre, post, and comb column j for each j) starts on an 8-sublane boundary, so all in-kernel
    slices are tile aligned (n_pad = 8 * (2 + hc) = 48 for hc=4).
  * ``mixT = fn_t . x^T`` (flash-attention style NT matmul) gives the projection with tokens on
    LANES: each gate group is one [8, bt] vreg-tile, the Sinkhorn row normalisation is plain
    element-wise math across the hc column tiles, and the column normalisation is an 8-row
    sublane reduction. That is ~(2 + 2*hc) single-tile ops per half iteration.
  * The per-token RMS factor is needed with tokens on lanes; it is read off the diagonal of the
    Gram matrix ``x . x^T`` (bf16 products are exact, f32 accumulation), so no transposes.
    Since rms is a per-token scalar it is applied after the projection (``mix = rms * (x.fn)``),
    which keeps the MXU operands in bf16 while staying f32-exact.
  * The finished [n_pad, bt] gate tile is turned back into token-major [bt, n_pad] with an
    identity matmul (HIGHEST precision) and written as the ``state`` output that the post kernel
    and the collapse consume as per-token [bt, 1] columns.

Partial edge blocks (T not a multiple of bt) are handled by masking garbage token lanes before
the identity matmul; per-token math never mixes tokens otherwise.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SLAB = 8  # sublane tile: each gate group occupies one 8-row slab of fn_t / mixT
_VMEM_LIMIT = 48 * 1024 * 1024


def _n_pad(hc: int) -> int:
    assert hc <= _SLAB, f"hc_mult={hc} > {_SLAB} not supported"
    return _SLAB * (2 + hc)


def _row_index(hc: int):
    """For each original mix column (pre[h], post[h], comb[i, j]) its row in the padded layout."""
    rows = [h for h in range(hc)]
    rows += [_SLAB + h for h in range(hc)]
    rows += [2 * _SLAB + j * _SLAB + i for i in range(hc) for j in range(hc)]  # comb[i, j]
    return jnp.asarray(rows, jnp.int32)


def prepare_fn_t(fn: jax.Array, hc: int) -> jax.Array:
    """fn [hc*D, (2+hc)*hc] -> fn_t [n_pad, hc*D] (group-aligned rows, zero padding)."""
    k, n = fn.shape
    assert n == (2 + hc) * hc, (fn.shape, hc)
    return jnp.zeros((_n_pad(hc), k), fn.dtype).at[_row_index(hc)].set(fn.T)


def prepare_params(scale: jax.Array, base: jax.Array, hc: int):
    """-> (scale_rows, base_rows), each [n_pad, 1] f32, matching prepare_fn_t's row layout."""
    scale = scale.astype(jnp.float32)
    seg = jnp.asarray([0] * hc + [1] * hc + [2] * (hc * hc), jnp.int32)
    rows = _row_index(hc)
    n_pad = _n_pad(hc)
    sc = jnp.zeros((n_pad,), jnp.float32).at[rows].set(scale[seg])
    bs = jnp.zeros((n_pad,), jnp.float32).at[rows].set(base.astype(jnp.float32))
    return sc.reshape(n_pad, 1), bs.reshape(n_pad, 1)


def unpack_state(state: jax.Array, hc: int):
    """state [T, n_pad] -> (pre [T, hc], post [T, hc], comb [T, hc, hc]) in f32."""
    rows = _row_index(hc)
    g = state[:, rows]
    return g[:, :hc], g[:, hc : 2 * hc], g[:, 2 * hc :].reshape(-1, hc, hc)


def _block_tokens(t: int, block_tokens: int | None) -> int:
    bt = int(block_tokens) if block_tokens else (32 if t <= 256 else 128)
    assert bt % 16 == 0, "block_tokens must be a multiple of 16 (bf16 sublane packing)"
    return t if t <= bt else bt  # a full-dim block is always a legal block shape


def _nt(a, b, precision=None):
    """a [M, K] . b [N, K]^T -> [M, N] f32."""
    return jax.lax.dot_general(
        a, b, (((1,), (1,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
    )


def _pre_kernel(x_ref, fnt_ref, sc_ref, bs_ref, col_ref, state_ref, *, hc, d, iters, hc_eps,
                eps, n_tokens, bt, precision):
    x = x_ref[...]
    if x.dtype != fnt_ref.dtype:
        x = x.astype(fnt_ref.dtype)
    k = x.shape[1]

    # RMS: diagonal of the Gram matrix (exact bf16 products, f32 accumulation).
    g = _nt(x, x, precision)  # [bt, bt]
    r_i = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 0)
    c_i = jax.lax.broadcasted_iota(jnp.int32, (bt, bt), 1)
    eye = r_i == c_i
    ssq = jnp.sum(jnp.where(eye, g, 0.0), axis=0, keepdims=True)  # [1, bt] (tokens on lanes)
    rinv = jax.lax.rsqrt(ssq * (1.0 / k) + eps)

    mix_t = _nt(fnt_ref[...], x, precision) * rinv  # [n_pad, bt]
    z = mix_t * sc_ref[...] + bs_ref[...]

    s = _SLAB
    pre_s = jax.nn.sigmoid(z[0:s]) + hc_eps
    post_s = 2.0 * jax.nn.sigmoid(z[s : 2 * s])
    # cols[j][i, t] = comb[t, i, j]; rows >= hc of every slab are padding.
    cols = [z[2 * s + j * s : 2 * s + (j + 1) * s] for j in range(hc)]

    m = cols[0]
    for j in range(1, hc):
        m = jnp.maximum(m, cols[j])
    cols = [jnp.exp(c - m) for c in cols]
    tot = cols[0]
    for j in range(1, hc):
        tot = tot + cols[j]
    cols = [c / tot + hc_eps for c in cols]

    row_ok = jax.lax.broadcasted_iota(jnp.int32, (s, bt), 0) < hc

    def col_norm(cs):
        return [c / (jnp.sum(jnp.where(row_ok, c, 0.0), axis=0, keepdims=True) + hc_eps)
                for c in cs]

    def row_norm(cs):
        rs = cs[0]
        for j in range(1, hc):
            rs = rs + cs[j]
        rs = rs + hc_eps
        return [c / rs for c in cs]

    cols = col_norm(cols)
    if iters > 1:
        def body(_, cs):
            return tuple(col_norm(row_norm(list(cs))))

        cols = list(jax.lax.fori_loop(0, iters - 1, body, tuple(cols), unroll=True))

    state_t = jnp.concatenate([pre_s, post_s, *cols], axis=0)  # [n_pad, bt]
    # Garbage lanes of a partial edge block must not reach the identity matmul (0 * NaN).
    n_valid = n_tokens - pl.program_id(0) * bt
    lane = jax.lax.broadcasted_iota(jnp.int32, state_t.shape, 1)
    state_t = jnp.where(lane < n_valid, state_t, 0.0)
    state = _nt(eye.astype(jnp.float32), state_t, jax.lax.Precision.HIGHEST)  # [bt, n_pad]
    state_ref[...] = state

    acc = state[:, 0:1] * x_ref[:, 0:d].astype(jnp.float32)
    for h in range(1, hc):
        acc = acc + state[:, h : h + 1] * x_ref[:, h * d : (h + 1) * d].astype(jnp.float32)
    col_ref[...] = acc.astype(col_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("hc", "hc_iters", "hc_eps", "eps", "block_tokens", "interpret"),
)
def glm5_mhc_pre(streams, fn, scale, base, *, hc: int, hc_iters: int, hc_eps: float,
                 eps: float, block_tokens: int | None = None, interpret: bool | None = None):
    """streams [T, hc*D], fn [hc*D, (2+hc)*hc], base [(2+hc)*hc], scale [3].

    Returns (collapsed [T, D] in streams.dtype, state [T, n_pad] f32); see ``unpack_state``.
    """
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    t, k = streams.shape
    d = k // hc
    assert d * hc == k and d % 128 == 0, (streams.shape, hc)
    n_pad = _n_pad(hc)
    bt = _block_tokens(t, block_tokens)

    if streams.dtype == jnp.bfloat16 and fn.dtype == jnp.bfloat16:
        precision, op_dtype = None, jnp.bfloat16  # exact products, f32 MXU accumulation
    else:
        precision, op_dtype = jax.lax.Precision.HIGHEST, jnp.float32
    fn_t = prepare_fn_t(fn.astype(op_dtype), hc)
    sc, bs = prepare_params(scale, base, hc)

    kernel = functools.partial(
        _pre_kernel, hc=hc, d=d, iters=int(hc_iters), hc_eps=float(hc_eps), eps=float(eps),
        n_tokens=t, bt=bt, precision=precision,
    )
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(t, bt),),
        in_specs=[
            pl.BlockSpec((bt, k), lambda i: (i, 0)),
            pl.BlockSpec((n_pad, k), lambda i: (0, 0)),
            pl.BlockSpec((n_pad, 1), lambda i: (0, 0)),
            pl.BlockSpec((n_pad, 1), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bt, d), lambda i: (i, 0)),
            pl.BlockSpec((bt, n_pad), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((t, d), streams.dtype),
            jax.ShapeDtypeStruct((t, n_pad), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT
        ),
        interpret=interpret,
        name="glm5-mhc-pre",
    )(streams, fn_t, sc, bs)


def _post_kernel(y_ref, x_ref, st_ref, o_ref, *, hc, d):
    st = st_ref[...]
    y = y_ref[...].astype(jnp.float32)
    for j in range(hc):
        acc = st[:, _SLAB + j : _SLAB + j + 1] * y
        for i in range(hc):
            c = 2 * _SLAB + j * _SLAB + i  # comb[i, j]
            acc = acc + st[:, c : c + 1] * x_ref[:, i * d : (i + 1) * d].astype(jnp.float32)
        o_ref[:, j * d : (j + 1) * d] = acc.astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("hc", "block_tokens", "interpret"))
def glm5_mhc_post(y, state, streams, *, hc: int, block_tokens: int | None = None,
                  interpret: bool | None = None):
    """y [T, D], state [T, n_pad] (from glm5_mhc_pre), streams [T, hc*D] -> [T, hc*D]."""
    if interpret is None:
        interpret = jax.default_backend() != "tpu"
    t, k = streams.shape
    d = k // hc
    assert y.shape == (t, d) and d % 128 == 0, (y.shape, streams.shape)
    n_pad = _n_pad(hc)
    assert state.shape == (t, n_pad), state.shape
    bt = _block_tokens(t, block_tokens)
    return pl.pallas_call(
        functools.partial(_post_kernel, hc=hc, d=d),
        grid=(pl.cdiv(t, bt),),
        in_specs=[
            pl.BlockSpec((bt, d), lambda i: (i, 0)),
            pl.BlockSpec((bt, k), lambda i: (i, 0)),
            pl.BlockSpec((bt, n_pad), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((bt, k), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((t, k), streams.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT
        ),
        interpret=interpret,
        name="glm5-mhc-post",
    )(y, streams, state.astype(jnp.float32))


__all__ = ["glm5_mhc_pre", "glm5_mhc_post", "prepare_fn_t", "prepare_params", "unpack_state"]
