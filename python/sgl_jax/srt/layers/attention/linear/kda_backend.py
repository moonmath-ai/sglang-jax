from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.kda import (
    chunk_kda,
    is_mega_kda_layout_supported,
    kda_forward_packed,
    naive_recurrent_kda,
)
from sgl_jax.srt.kernels.kda.chain_verify import kda_chain_verify
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.profiling_utils import named_scope


# Conv pool layout (lane-dense, see RecurrentStatePool conv_tiled): [slots, K-1, 3, H, D] with
# the q/k/v streams on axis 2 and heads sharded over "tensor". The short-conv kernels take
# one stream at a time as [B, H*D, K-1].
_CONV_POOL_SPEC = P("data", None, None, "tensor", None)
_CONV_STREAM_SPEC = P("data", "tensor", None)


def _conv_pool_to_streams(x):
    """Local [B, K-1, 3, H, D] -> three [B, H*D, K-1]."""
    B, km1 = x.shape[:2]
    return tuple(jnp.swapaxes(x[:, :, s].reshape(B, km1, -1), 1, 2) for s in range(3))


def _conv_streams_to_pool(streams, head_dim):
    """Inverse of :func:`_conv_pool_to_streams`."""
    *lead, hd, km1 = streams[0].shape
    return jnp.stack(
        [jnp.swapaxes(x, -1, -2).reshape(*lead, km1, hd // head_dim, head_dim) for x in streams],
        axis=-3,
    )


if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def _gate_lower_bound(layer):
    """KDA gate lower bound (``None``: softplus gate), under either attribute name."""
    lower_bound = getattr(layer, "kda_gate_lower_bound", None)
    return getattr(layer, "kda_lower_bound", None) if lower_bound is None else lower_bound


def l2_normalize(x: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """Match FLA ``l2norm_fwd``: ``x / sqrt(sum(x**2) + eps)``."""
    x32 = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + epsilon)
    return (x32 * rstd).astype(x.dtype)


class KDAAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, mesh: jax.sharding.Mesh = None):
        super().__init__(
            mesh=mesh,
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        a: jax.Array,
        b: jax.Array,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool: RecurrentStatePool,
        **kwargs,
    ) -> jax.Array:
        recurrent_indices = self.forward_metadata.recurrent_indices
        if forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:
            # The fused verify kernel reads state + conv history from the pool by slot itself.
            return self._forward_target_verify(
                q, k, v, a, b, layer, recurrent_state_pool, recurrent_indices
            )
        ssm_states, conv_states = self.get_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices
        )
        q_conv_w = layer.q_conv1d.weight.value
        k_conv_w = layer.k_conv1d.weight.value
        v_conv_w = layer.v_conv1d.weight.value
        q_state, k_state, v_state = conv_states

        cu_q_lens = self.forward_metadata.cu_q_lens
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            q, q_state_new = self._short_conv_extend(
                q, q_conv_w, q_state, cu_q_lens, layer.activation
            )
            k, k_state_new = self._short_conv_extend(
                k, k_conv_w, k_state, cu_q_lens, layer.activation
            )
            v, v_state_new = self._short_conv_extend(
                v, v_conv_w, v_state, cu_q_lens, layer.activation
            )
        else:
            q, q_state_new = short_convolution(
                q,
                q_conv_w,
                q_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
            k, k_state_new = short_convolution(
                k,
                k_conv_w,
                k_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
            v, v_state_new = short_convolution(
                v,
                v_conv_w,
                v_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
        new_conv = (q_state_new, k_state_new, v_state_new)

        q = q.reshape(q.shape[0], layer.num_q_heads, layer.head_q_dim)
        k = k.reshape(k.shape[0], layer.num_k_heads, layer.head_k_dim)
        v = v.reshape(v.shape[0], layer.num_v_heads, layer.head_v_dim)
        a = a.reshape(a.shape[0], layer.num_q_heads, layer.head_q_dim)
        b = b.reshape(b.shape[0], layer.num_q_heads)

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._forward_extend(
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                cu_q_lens,
                layer,
                scale=layer.scale,
            )
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            # Decode uses the recurrent JAX path, so normalize Q/K before the
            # call. Mega KDA performs the same normalization inside its prefill
            # kernel; the chunked prefill fallback normalizes in its branch.
            q = l2_normalize(q)
            k = l2_normalize(k)
            output, new_recurrent = self._forward_decode(
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                layer,
                scale=layer.scale,
            )
        else:
            raise NotImplementedError(f"KDA does not support {forward_batch.forward_mode}")

        new_ssm_full = self.set_ssm_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_recurrent
        )
        new_conv_full_list = self.set_conv_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_conv
        )

        # Track snapshot: boundaries are forced to forward ends, so the
        # running-scatter value IS the boundary snapshot; scatter it into each
        # request's track slot too. None -> byte-identical no-track path.
        track_indices = self.forward_metadata.recurrent_track_indices
        if track_indices is not None:
            track_mask = self.forward_metadata.recurrent_track_mask
            new_ssm_full = self.set_ssm_track_state(
                new_ssm_full, track_indices, track_mask, new_recurrent
            )
            new_conv_full_list = self.set_conv_track_state(
                new_conv_full_list, track_indices, track_mask, new_conv
            )

        return output.reshape(output.shape[0], -1), (new_ssm_full, new_conv_full_list)

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices):
        """Return per-request views of (ssm, conv) state for this layer."""
        recurrent_buffer, conv_buffer_list = self.get_layer_cache(
            recurrent_state_pool,
            layer_id,
        )
        # [conv] or [conv, spec_conv, spec_ssm] (speculative verify scratch).
        assert len(conv_buffer_list) in (1, 3), (
            f"KDA expects 1 conv buffer (+2 speculative scratch) per layer; "
            f"got {len(conv_buffer_list)}"
        )
        conv_buffer = conv_buffer_list[0]

        ssm = jax.shard_map(
            lambda buf, idx: buf[idx],
            mesh=self.mesh,
            in_specs=(P("data", "tensor", None, None), P("data")),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(recurrent_buffer, recurrent_indices)

        assert conv_buffer.ndim == 5, "KDA needs the tiled conv layout (conv_tiled)"
        conv = jax.shard_map(
            lambda buf, idx: _conv_pool_to_streams(buf[idx]),
            mesh=self.mesh,
            in_specs=(_CONV_POOL_SPEC, P("data")),
            out_specs=(_CONV_STREAM_SPEC,) * 3,
            check_vma=False,
        )(conv_buffer, recurrent_indices)

        has_initial_state = self.forward_metadata.has_initial_state
        if has_initial_state is not None:
            ssm = jnp.where(has_initial_state[:, None, None, None], ssm, 0.0)
            conv = tuple(jnp.where(has_initial_state[:, None, None], c, 0.0) for c in conv)

        return ssm, conv

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        """Scatter per-request ``new_recurrent`` into the FULL pool buffer.

        Suppress writes at idx==0: padding rows carry idx=0 and would otherwise
        pollute the per-rank dummy slot, leaking garbage back as initial state.
        """
        full_recurrent, _ = self.get_layer_cache(recurrent_state_pool, layer_id)

        def _scatter(buf, idx, val):
            keep_mask = (idx == 0).reshape(-1, 1, 1, 1)
            safe_val = jnp.where(keep_mask, buf[idx], val)
            return buf.at[idx].set(safe_val)

        return jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
                P("data", "tensor", None, None),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(full_recurrent, recurrent_indices, new_recurrent)

    def _scatter_conv(self, full_conv, dst, keep, new_conv):
        """``full_conv[dst] = new_conv`` (per-stream q/k/v states) except rows where ``keep``."""
        hd = full_conv.shape[-1]

        def _scatter(buf, d, kp, *streams):
            val = _conv_streams_to_pool(streams, hd)
            kp = kp.reshape((-1,) + (1,) * (buf.ndim - 1))
            return buf.at[d].set(jnp.where(kp, buf[d], val.astype(buf.dtype)))

        return jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(_CONV_POOL_SPEC, P("data"), P("data")) + (_CONV_STREAM_SPEC,) * 3,
            out_specs=_CONV_POOL_SPEC,
            check_vma=False,
        )(full_conv, dst, keep, *new_conv)

    def set_conv_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_conv):
        """Scatter per-request conv state. Same idx==0 guard as set_ssm_state."""
        _, conv_buffer_list = self.get_layer_cache(recurrent_state_pool, layer_id)
        new_conv_full = self._scatter_conv(
            conv_buffer_list[0], recurrent_indices, recurrent_indices == 0, new_conv
        )
        return [new_conv_full] + list(conv_buffer_list[1:])

    def set_ssm_track_state(self, full_recurrent, track_indices, track_mask, new_recurrent):
        """Scatter ``new_recurrent`` into the request track slots.

        Operates on the buffer RETURNED by ``set_ssm_state`` so both the
        running slot and the track slot persist. Keep-mask preserves slots at
        ``track_idx == 0`` (padding/dummy) and ``track_mask == 0`` (non-boundary
        rows). Wrapped in ``optimization_barrier`` because the buffer is donated
        (donated-buffer multi-host aliasing guard).
        """

        def _scatter(buf, tidx, tmask, val):
            keep = ((tidx == 0) | (tmask == 0)).reshape(-1, 1, 1, 1)
            safe_val = jnp.where(keep, buf[tidx], val)
            return buf.at[tidx].set(safe_val)

        new_full = jax.shard_map(
            _scatter,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
                P("data"),
                P("data", "tensor", None, None),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(full_recurrent, track_indices, track_mask, new_recurrent)
        return jax.lax.optimization_barrier(new_full)

    def set_conv_track_state(self, new_conv_full_list, track_indices, track_mask, new_conv):
        """Scatter ``new_conv`` into the request track slots.

        Conv variant of :meth:`set_ssm_track_state`.
        """
        new_full = self._scatter_conv(
            new_conv_full_list[0],
            track_indices,
            (track_indices == 0) | (track_mask == 0),
            new_conv,
        )
        return [jax.lax.optimization_barrier(new_full)] + list(new_conv_full_list[1:])

    # ------------------------------------------------------------------
    # Forward mode implementations
    # ------------------------------------------------------------------

    def _short_conv_extend(
        self,
        x: jax.Array,
        weight: jax.Array,
        cache: jax.Array,
        cu_seqlens: jax.Array,
        activation,
    ) -> tuple[jax.Array, jax.Array]:
        """EXTEND-path conv wrapped in shard_map."""

        def _call(x, weight, cache, cu_seqlens):
            return short_convolution(
                x, weight, cache, cu_seqlens, ForwardMode.EXTEND, activation=activation
            )

        sharded = jax.shard_map(
            _call,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor"),  # x [T, hidden]
                P("tensor", None),  # weight [hidden, K]
                P("data", "tensor", None),  # cache [B, hidden, K-1]
                P("data"),  # cu_seqlens [B+1]
            ),
            out_specs=(
                P("data", "tensor"),  # y [T, hidden]
                P("data", "tensor", None),  # new_cache [B, hidden, K-1]
            ),
            check_vma=False,
        )
        return sharded(x, weight, cache, cu_seqlens)

    @named_scope("kda_extend")
    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Prefill via Mega KDA, with a guarded fallback to the chunked kernel."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = q.shape[-2]
        # kda_gate_chunk_cumsum requires A_log shape (H,); layer stores [1,1,H,1].
        A_log = layer.A_log.value.reshape(H)
        dt_bias = layer.dt_bias.value.reshape(H, -1)
        scale = scale if scale is not None else layer.scale
        lower_bound = _gate_lower_bound(layer)
        kernel = os.environ.get("SGLANG_JAX_KDA_PREFILL_KERNEL", "mega").strip().lower()
        if kernel not in {"chunked", "mega"}:
            raise ValueError(
                f"SGLANG_JAX_KDA_PREFILL_KERNEL must be 'chunked' or 'mega', got {kernel!r}"
            )
        use_mega = kernel == "mega"

        def _prefill_call(q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias):
            operands = (q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)

            def _chunked(args):
                q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = args
                q = l2_normalize(q)
                k = l2_normalize(k)
                o, final_state, *_ = chunk_kda(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    use_gate_in_kernel=True,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                )
                return o, final_state

            if not use_mega:
                return _chunked(operands)

            def _mega(args):
                q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = args
                return kda_forward_packed(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu_seqlens=cu_seqlens,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=scale,
                    initial_state=initial_state,
                    lower_bound=lower_bound,
                )

            padded_tokens = (q.shape[1] + 63) // 64 * 64
            layout_supported = is_mega_kda_layout_supported(
                cu_seqlens,
                padded_tokens,
            )
            return jax.lax.cond(layout_supported, _mega, _chunked, operands)

        sharded = jax.shard_map(
            _prefill_call,
            mesh=self.mesh,
            in_specs=(
                P(None, "data", "tensor", None),  # q [1, T, H, K]
                P(None, "data", "tensor", None),  # k [1, T, H, K]
                P(None, "data", "tensor", None),  # v [1, T, H, V]
                P(None, "data", "tensor", None),  # g [1, T, H, K]
                P(None, "data", "tensor"),  # beta [1, T, H]
                P("data", "tensor", None, None),  # initial_state [N, H, K, V]
                P("data"),  # cu_seqlens [N+1]
                P("tensor"),  # A_log [H]
                P("tensor", None),  # dt_bias [H, K]
            ),
            out_specs=(
                P(None, "data", "tensor", None),  # output [1, T, H, V]
                P("data", "tensor", None, None),  # final_state [N, H, K, V]
            ),
            check_vma=False,
        )
        # Kernel expects [1, T_packed, H, K] packed layout; strip after.
        o, final_state = sharded(
            q[None, ...],
            k[None, ...],
            v[None, ...],
            g[None, ...],
            beta[None, ...],
            initial_state,
            cu_seqlens,
            A_log,
            dt_bias,
        )
        return o[0], final_state

    @named_scope("kda_decode")
    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode via naive JAX recurrence (Pallas decode TBD)."""
        g = self._fused_kda_gate(layer, g)

        def _decode_kernel(q_d, k_d, v_d, g_d, beta_d, h0):
            o, final_state = naive_recurrent_kda(
                q_d[:, None, ...],
                k_d[:, None, ...],
                v_d[:, None, ...],
                g_d[:, None, ...],
                beta_d[:, None, ...],
                scale=scale,
                initial_state=h0,
                output_final_state=True,
            )
            return o[:, 0], final_state

        sharded = jax.shard_map(
            _decode_kernel,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),  # q [B, H, K]
                P("data", "tensor", None),  # k [B, H, K]
                P("data", "tensor", None),  # v [B, H, V]
                P("data", "tensor", None),  # g [B, H, K]
                P("data", "tensor"),  # beta [B, H]
                P("data", "tensor", None, None),  # h0 [B, H, K, V]
            ),
            out_specs=(
                P("data", "tensor", None),  # o [B, H, V]
                P("data", "tensor", None, None),  # final_state [B, H, K, V]
            ),
            check_vma=False,
        )
        return sharded(q, k, v, g, beta, initial_state)

    @named_scope("kda_target_verify")
    def _forward_target_verify(self, q, k, v, a, b, layer, recurrent_state_pool, recurrent_indices):
        """Speculative target verify over a topk=1 chain of T tokens per request.

        Tokens are request-major (``[bs * T]``). One Pallas kernel per layer runs conv + gate +
        L2 norm + recurrence from the pre-conv projections, reading the slot's conv history and
        state straight from the pool and writing the state (and conv window) after every chain
        step into the pool's verify scratch; the running state is NOT advanced here
        (``RecurrentStatePool.commit_speculative_state`` keeps the accepted step).
        """
        ssm_full, conv_list = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
        assert len(conv_list) == 3, (
            "TARGET_VERIFY needs the recurrent pool's speculative scratch "
            "(RecurrentStatePool(spec_num_draft_tokens=...))"
        )
        conv_full, spec_conv_full, spec_ssm_full = conv_list
        bs = recurrent_indices.shape[0]
        T = q.shape[0] // bs
        assert T * bs == q.shape[0], (q.shape, bs)
        Kd, Vd = layer.head_q_dim, layer.head_v_dim
        assert Kd == Vd, (Kd, Vd)
        lower_bound = _gate_lower_bound(layer)
        scale = layer.scale
        q_w, k_w, v_w = (c.weight.value for c in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d))

        conv_w_tiled = getattr(layer, "conv_w_tiled", None)
        if conv_w_tiled is not None:  # prepared at load: [W, 3, H, Kd]
            cw_args, cw_specs = (conv_w_tiled.value,), (P(None, None, "tensor", None),)
        else:
            cw_args, cw_specs = (q_w, k_w, v_w), (P("tensor", None),) * 3

        def _local(q_d, k_d, v_d, a_d, b_d, a_log, dtb, *rest):
            n_cw = len(cw_args)
            cws, (cpool, pool, idx, st, stc) = rest[:n_cw], rest[n_cw:]
            B = idx.shape[0]
            Hl = a_log.size
            if n_cw == 1:
                cw = cws[0].astype(q_d.dtype)
            else:  # [D_loc, W] x3 -> [W, 3, Hl, Kd]
                cw = jnp.stack(cws, axis=0).reshape(3, Hl, Kd, -1)
                cw = jnp.transpose(cw, (3, 0, 1, 2)).astype(q_d.dtype)
            r4 = lambda x, d: x.reshape(B, T, Hl, d)
            o, st, stc = kda_chain_verify(
                r4(q_d, Kd),
                r4(k_d, Kd),
                r4(v_d, Vd),
                r4(a_d, Kd),
                b_d.reshape(B, T, Hl),
                a_log.reshape(Hl),
                dtb.reshape(Hl, Kd),
                cw,
                cpool,
                pool,
                idx,
                st,
                stc,
                scale=scale,
                lower_bound=lower_bound,
            )
            return o.reshape(B * T, Hl * Vd), st, stc

        spec_ssm_spec = P("data", None, "tensor", None, None)
        spec_conv_spec = P("data", None, None, None, "tensor", None)
        output, new_spec_ssm, new_spec_conv = jax.shard_map(
            _local,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor"),  # q [B*T, H*K] pre-conv
                P("data", "tensor"),  # k
                P("data", "tensor"),  # v
                P("data", "tensor", None),  # raw gate [B*T, H, K]
                P("data", "tensor"),  # beta [B*T, H]
                P(None, None, "tensor", None),  # A_log [1, 1, H, 1]
                P("tensor"),  # dt_bias [H*K]
                *cw_specs,  # conv taps
                _CONV_POOL_SPEC,
                P("data", "tensor", None, None),  # ssm pool
                P("data"),  # slot indices
                spec_ssm_spec,
                spec_conv_spec,
            ),
            out_specs=(P("data", "tensor"), spec_ssm_spec, spec_conv_spec),
            check_vma=False,
        )(
            q,
            k,
            v,
            a.reshape(bs * T, layer.num_q_heads, layer.head_q_dim),
            b.reshape(bs * T, layer.num_q_heads),
            layer.A_log.value,
            layer.dt_bias.value,
            *cw_args,
            conv_full,
            ssm_full,
            recurrent_indices,
            spec_ssm_full,
            spec_conv_full,
        )
        return output.reshape(output.shape[0], -1), (
            ssm_full,
            [conv_full, new_spec_conv, new_spec_ssm],
        )

    def _fused_kda_gate(
        self,
        layer: RadixLinearAttention,
        g: jax.Array,
    ) -> jax.Array:
        """JAX-side gate activation used by the DECODE path."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = g.shape[-2]
        orig_dtype = g.dtype
        g32 = g.astype(jnp.float32) + layer.dt_bias.value.reshape(H, -1).astype(jnp.float32)
        a_scale = jnp.exp(layer.A_log.value.reshape(H, 1).astype(jnp.float32))
        lower_bound = _gate_lower_bound(layer)
        if lower_bound is None:
            out = -a_scale * jax.nn.softplus(g32)
        else:
            out = lower_bound * jax.nn.sigmoid(a_scale * g32)
        return out.astype(orig_dtype)


__all__ = ["KDAAttnBackend"]
