"""Short depthwise causal conv1d used by linear-attention backends (e.g. KDA).

The convolution is intentionally implemented as a stateless function (not an
nnx Module) so backends can freely combine it with their own weight containers
and cache layouts. Two execution paths are provided:

* ``decode`` — single-token step that appends the new token to a per-sequence
  ``[B, D, K-1]`` cache, runs the conv on the resulting K-token window, and
  drops the oldest slot before writing back.
* ``extend`` — variable-length packed prefill that consumes ``cu_seqlens`` to
  build a per-token sliding window mixing prior cache and in-sequence tokens.

State convention follows vLLM: cache has width ``K-1`` and stores the
prior ``K-1`` tokens (the current token is supplied via ``x`` at call time).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.profiling_utils import named_scope

# Map of supported activation names → callable. ``None`` means identity.
_ACTIVATION_FNS: dict[str | None, Callable[[jax.Array], jax.Array] | None] = {
    None: None,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
}


def _resolve_activation(
    activation: str | Callable[[jax.Array], jax.Array] | None,
) -> Callable[[jax.Array], jax.Array] | None:
    """Resolve an activation spec to a callable (or None for identity).

    Accepts either a name from ``_ACTIVATION_FNS`` or a user-supplied callable.
    """
    if activation is None or callable(activation):
        return activation
    if activation not in _ACTIVATION_FNS:
        raise ValueError(
            f"short_convolution activation must be one of {sorted(k for k in _ACTIVATION_FNS if k is not None)} "
            f"or a callable; got {activation!r}"
        )
    return _ACTIVATION_FNS[activation]


def short_convolution(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array | None,
    forward_mode: ForwardMode,
    bias: jax.Array | None = None,
    activation: str | Callable[[jax.Array], jax.Array] | None = "silu",
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d with per-sequence cache.

    Args:
        x: ``[T, D]`` for ``EXTEND`` (packed varlen) or ``[B, D]`` for ``DECODE``.
        weight: depthwise kernel ``[D, K]``.
        cache: per-sequence rolling buffer ``[B, D, K-1]`` storing the prior
            ``K-1`` tokens (zeros for fresh sequences). The current token is
            supplied via ``x`` and not written into the input cache.
        cu_seqlens: ``[N+1]`` cumulative sequence lengths; required for
            ``EXTEND``, ignored for ``DECODE``.
        forward_mode: ``ForwardMode.DECODE`` or ``ForwardMode.EXTEND``.
        bias: optional ``[D]`` channel bias added before the activation.
        activation: name (e.g. ``"silu"``, ``"gelu"``, ``"sigmoid"``), a
            user-supplied callable, or ``None`` for identity.

    Returns:
        ``(y, new_cache)`` where ``y`` matches the leading dims of ``x`` and
        ``new_cache`` has the same shape as ``cache``.
    """
    activation_fn = _resolve_activation(activation)

    weight = _normalize_weight(weight)

    if forward_mode == ForwardMode.DECODE:
        return _decode_conv(x, weight, cache, bias, activation_fn)
    if cu_seqlens is None:
        raise ValueError("short_convolution(EXTEND) requires cu_seqlens")
    return _extend_conv(x, weight, cache, cu_seqlens, bias, activation_fn)


def _normalize_weight(weight: jax.Array) -> jax.Array:
    """Reduce common conv-weight layouts to ``[D, K]``."""
    # Squeeze the depthwise singleton axis if the loader handed us [D, 1, K].
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0, :]
    return weight


def _apply_activation(
    y: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> jax.Array:
    if activation_fn is None:
        return y
    return activation_fn(y)


@named_scope("short_conv_decode")
def _decode_conv(
    x: jax.Array,  # [B, D]
    conv_kernel: jax.Array,  # [D, K]
    cache: jax.Array,  # [B, D, K-1]
    bias: jax.Array | None,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    # expand x shape from [B, D] to [B, D, 1]
    new_cache = jnp.concatenate([cache, x[..., None]], axis=-1)
    y = jnp.einsum("bck,ck->bc", new_cache, conv_kernel.astype(new_cache.dtype))
    if bias is not None:
        y = y + bias.astype(y.dtype)
    y = _apply_activation(y, activation_fn)
    # return the last K-1 conv state
    return y, new_cache[:, :, 1:]


@named_scope("short_conv_extend")
def _extend_conv(
    x: jax.Array,  # [T, D]
    conv_kernel: jax.Array,  # [D, K]
    cache: jax.Array,  # [B, D, K-1]
    cu_seqlens: jax.Array,
    bias: jax.Array | None,
    activation_fn: Callable[[jax.Array], jax.Array] | None,
) -> tuple[jax.Array, jax.Array]:
    T = x.shape[0]
    K = conv_kernel.shape[-1]
    W = K - 1  # cache width

    # Locate every output token within its owning sequence.
    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    starts = cu_seqlens[:-1][seq_ids]

    # Varlen causal depthwise conv as K shifted multiply-adds (no [T, D, K] window gather):
    # tap k reads x shifted by s = K-1-k rows, masked where the source row belongs to an
    # earlier sequence.
    D = x.shape[1]
    B = cache.shape[0]
    w = conv_kernel.astype(jnp.float32)  # [D, K]
    acc = jnp.zeros((T, D), jnp.float32)
    for k in range(K):
        s = K - 1 - k
        xs = x if s == 0 else jnp.concatenate([jnp.zeros((s, D), x.dtype), x[: T - s]], axis=0)
        in_seq = (token_idx - s) >= starts
        acc = acc + jnp.where(in_seq[:, None], xs.astype(jnp.float32), 0.0) * w[:, k]
    if W > 0:
        # The prior cache (slots [0, W-1] = the W tokens before the sequence) only reaches
        # the first W rows of each sequence: row start_b + i (i < W) sees cache slot
        # W + i - s for lags s in (i, W]. Computed on [B, D] per row and scatter-added.
        cache32 = cache.astype(jnp.float32)  # [B, D, W]
        seq_starts = cu_seqlens[:-1]
        seq_lens = cu_seqlens[1:] - seq_starts
        for i in range(W):
            contrib = jnp.zeros((B, D), jnp.float32)
            for s in range(i + 1, W + 1):
                contrib = contrib + cache32[:, :, W + i - s] * w[:, K - 1 - s]
            rows = jnp.where(i < seq_lens, seq_starts + i, T)
            acc = acc.at[rows].add(contrib, mode="drop")
    y = acc.astype(x.dtype)
    if bias is not None:
        y = y + bias.astype(y.dtype)
    y = _apply_activation(y, activation_fn)

    # Compute the new per-sequence cache: the last W = K-1 input tokens of
    # each sequence, falling back to the prior cache when the sequence is
    # shorter than W.
    ends = cu_seqlens[1:]
    state_offsets = jnp.arange(W, dtype=cu_seqlens.dtype)
    final_idx = ends[:, None] - W + state_offsets[None, :]
    final_from_x = final_idx >= cu_seqlens[:-1, None]
    safe_final_idx = jnp.clip(final_idx, 0, jnp.maximum(T - 1, 0))
    final_x = jnp.swapaxes(x[safe_final_idx], 1, 2)
    final_cache_pos = jnp.clip(W + final_idx - cu_seqlens[:-1, None], 0, W - 1)
    final_cache = jnp.take_along_axis(cache, final_cache_pos[:, None, :], axis=2)
    new_cache = jnp.where(final_from_x[:, None, :], final_x, final_cache)

    return y, new_cache


__all__ = ["short_convolution"]
