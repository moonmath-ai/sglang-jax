"""RecurrentStatePool -- buffer pool for linear recurrent layers (KDA/GDN)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

# Module-level cache for jitted zero-allocators (recurrent/conv buffers).
_RECURRENT_ZERO_ALLOCATOR_CACHE: dict = {}


def _get_recurrent_zero_allocator(shape, dtype, sharding):
    """Return a cached jax.jit(jnp.zeros) allocator for recurrent/conv buffers."""
    key = (
        id(sharding.mesh),
        tuple(shape),
        str(jnp.dtype(dtype)),
        repr(sharding.spec),
        getattr(sharding, "memory_kind", None),
    )
    if key not in _RECURRENT_ZERO_ALLOCATOR_CACHE:
        _RECURRENT_ZERO_ALLOCATOR_CACHE[key] = jax.jit(
            partial(jnp.zeros, shape=tuple(shape), dtype=dtype),
            out_shardings=sharding,
        )
    return _RECURRENT_ZERO_ALLOCATOR_CACHE[key]


_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _resolve_dtype(env_var: str, default):
    name = os.environ.get(env_var)
    return _DTYPE_MAP[name] if name else default


@dataclass(frozen=True)
class RecurrentStateDType:
    conv: jnp.dtype
    temporal: jnp.dtype


@dataclass(frozen=True)
class LinearRecurrentStateParams:
    layers: list[int]
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    dtype: RecurrentStateDType
    # GDN has asymmetric K vs V projection widths (e.g.
    # Qwen3.5 GDN: num_k_heads=16/head_k_dim=128 vs num_v_heads=32/head_v_dim=128).
    # When None (KDA / Lightning / Bailing), RecurrentStatePool falls back to
    # treating K dim = V dim.
    num_k_heads: int | None = None
    head_k_dim: int | None = None


def recurrent_state_dtype() -> RecurrentStateDType:
    return RecurrentStateDType(
        conv=_resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16),
        temporal=_resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32),
    )


@register_pytree_node_class
class RecurrentStatePool:

    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        mesh: Mesh,
        dp_size: int = 1,
        recurrent_partition_axis: str = "tensor",
        conv_partition_axis: str = "tensor",
        data_partition_axis: str = "data",
        temporal_dtype=None,
        conv_dtype=None,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        abstract: bool = False,
        spec_num_draft_tokens: int = 0,
        conv_tiled: bool = False,
        lazy_commit: bool = False,
    ):
        """`size` is the **global** number of valid slots across all DP ranks
        (mirrors MHATokenToKVPool.size semantics). Internally we partition by
        DP: each rank gets `size // dp_size` valid slots + 1 dummy slot at
        index 0, so total buffer slots = size + dp_size.

        ``spec_num_draft_tokens > 0`` (speculative decoding, topk == 1 chain) adds per-slot
        verify scratch to every layer: target verify writes the state after each of its
        ``spec_num_draft_tokens`` tokens there instead of advancing the running state, and
        :meth:`commit_speculative_state` then keeps the step matching the accepted length.
        Scratch rows are *batch positions* of the verify batch (``total_slots`` rows bound
        the verify batch size). It rides in the layer's conv list as
        ``[conv, spec_conv [rows, T, proj, K-1], spec_ssm [rows, T, H, K, V]]`` so the model
        and backends pass it through untouched.

        ``conv_tiled`` (KDA, equal q/k/v widths): conv state stored lane-dense as
        ``[slots, K-1, 3, H, head_dim]`` (taps major, each tap a native [heads, 128] tile;
        spec_conv ``[rows, T, K-1, 3, H, head_dim]``) instead of ``[slots, proj, K-1]``, whose
        minor dim of K-1 forces relayouts around every use.

        ``lazy_commit`` (KDA conv-fused verify, radix cache off): the verify scratch is indexed
        by **slot** and each layer's conv list gets a 4th entry ``spec_step [slots] int32``:
        -1 = the slot's running state is in the pool, t >= 0 = it is scratch step t. Commit
        only records ``accepted - 1`` (no state copy); extend/decode write the pool and reset
        the slot to -1; verify seeds from whichever holds the state.
        """
        state_dtype = recurrent_state_dtype()
        if temporal_dtype is None:
            temporal_dtype = state_dtype.temporal
        if conv_dtype is None:
            conv_dtype = state_dtype.conv
        self.temporal_dtype = temporal_dtype
        self.conv_dtype = conv_dtype

        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        assert len(set(linear_recurrent_layer_ids)) == len(linear_recurrent_layer_ids), (
            f"linear_recurrent_layer_ids must not contain duplicates, "
            f"got {linear_recurrent_layer_ids}"
        )
        self.linear_recurrent_layer_ids: list[int] = list(linear_recurrent_layer_ids)
        self.layers_mapping: dict[int, int] = {
            layer_id: idx for idx, layer_id in enumerate(self.linear_recurrent_layer_ids)
        }
        self.num_linear_recurrent_layers: int = len(self.linear_recurrent_layer_ids)

        assert (
            size % dp_size == 0
        ), f"RecurrentStatePool size ({size}) must be divisible by dp_size ({dp_size})."

        self.size = size
        self.dp_size = dp_size
        self.slots_per_rank = size // dp_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.conv_kernel_size = conv_kernel_size

        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        self.proj_size = proj_v + 2 * proj_k

        # Each rank reserves slot 0 as a dummy → +1 per rank.
        self.total_slots = size + dp_size

        self.mesh = mesh
        self.recurrent_partition_axis = recurrent_partition_axis
        self.conv_partition_axis = conv_partition_axis
        self.data_partition_axis = data_partition_axis

        recurrent_axis_size = mesh.shape[recurrent_partition_axis]
        conv_axis_size = mesh.shape[conv_partition_axis]
        assert num_heads % recurrent_axis_size == 0, (
            f"num_heads {num_heads} must be divisible by "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert num_k_heads % recurrent_axis_size == 0, (
            f"num_k_heads {num_k_heads} must be divisible by "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert self.proj_size % conv_axis_size == 0, (
            f"proj_size {self.proj_size} must be divisible by "
            f"'{conv_partition_axis}' size {conv_axis_size}"
        )

        self.recurrent_sharding = NamedSharding(
            mesh, P(data_partition_axis, recurrent_partition_axis, None, None)
        )
        self.spec_num_draft_tokens = spec_num_draft_tokens
        self.conv_tiled = bool(conv_tiled)
        self.lazy_commit = bool(lazy_commit) and spec_num_draft_tokens > 0
        if self.conv_tiled:
            assert num_k_heads == num_heads and head_k_dim == head_dim, "tiled conv: equal q/k/v"
            self.conv_sharding = NamedSharding(
                mesh, P(data_partition_axis, None, None, conv_partition_axis, None)
            )
        else:
            self.conv_sharding = NamedSharding(
                mesh, P(data_partition_axis, conv_partition_axis, None)
            )

        self.recurrent_buffers, self.conv_buffers = self._create_buffers(abstract=abstract)

    def _create_buffers(self, *, abstract: bool = False) -> tuple[list, list]:
        recurrent_shape = (self.total_slots, self.num_heads, self.head_dim, self.head_dim)
        if self.conv_tiled:
            conv_shape = (
                self.total_slots, self.conv_kernel_size - 1, 3, self.num_heads, self.head_dim
            )
        else:
            conv_shape = (self.total_slots, self.proj_size, self.conv_kernel_size - 1)
        temporal_dtype = self.temporal_dtype
        conv_dtype = self.conv_dtype

        if abstract:
            conv_inner = [jax.ShapeDtypeStruct(conv_shape, conv_dtype, sharding=self.conv_sharding)]
            if self.spec_num_draft_tokens > 0:
                T = self.spec_num_draft_tokens
                conv_inner += [
                    jax.ShapeDtypeStruct(
                        (self.total_slots, T) + conv_shape[1:],
                        conv_dtype,
                        sharding=self.spec_conv_sharding,
                    ),
                    jax.ShapeDtypeStruct(
                        (self.total_slots, T) + recurrent_shape[1:],
                        temporal_dtype,
                        sharding=self.spec_recurrent_sharding,
                    ),
                ]
            return (
                [
                    jax.ShapeDtypeStruct(
                        recurrent_shape, temporal_dtype, sharding=self.recurrent_sharding
                    )
                    for _ in range(self.num_linear_recurrent_layers)
                ],
                [list(conv_inner) for _ in range(self.num_linear_recurrent_layers)],
            )

        alloc_recurrent = _get_recurrent_zero_allocator(
            recurrent_shape, temporal_dtype, self.recurrent_sharding
        )
        alloc_conv = _get_recurrent_zero_allocator(conv_shape, conv_dtype, self.conv_sharding)
        with jax.set_mesh(self.mesh):
            recurrent_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                recurrent_buffers.append(alloc_recurrent())

            conv_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                inner = []
                inner.append(alloc_conv())
                conv_buffers.append(inner)

            if self.spec_num_draft_tokens > 0:
                T = self.spec_num_draft_tokens
                alloc_spec_conv = _get_recurrent_zero_allocator(
                    (self.total_slots, T) + conv_shape[1:], conv_dtype, self.spec_conv_sharding
                )
                alloc_spec_ssm = _get_recurrent_zero_allocator(
                    (self.total_slots, T) + recurrent_shape[1:],
                    temporal_dtype,
                    self.spec_recurrent_sharding,
                )
                for inner in conv_buffers:
                    inner.append(alloc_spec_conv())
                    inner.append(alloc_spec_ssm())
                if self.lazy_commit:
                    step_sharding = NamedSharding(self.mesh, P(self.data_partition_axis))
                    for inner in conv_buffers:
                        inner.append(
                            jax.device_put(
                                jnp.full((self.total_slots,), -1, jnp.int32), step_sharding
                            )
                        )

        return recurrent_buffers, conv_buffers

    @property
    def spec_conv_sharding(self):
        if self.conv_tiled:
            return NamedSharding(
                self.mesh,
                P(self.data_partition_axis, None, None, None, self.conv_partition_axis, None),
            )
        return NamedSharding(
            self.mesh, P(self.data_partition_axis, None, self.conv_partition_axis, None)
        )

    @property
    def spec_recurrent_sharding(self):
        return NamedSharding(
            self.mesh,
            P(self.data_partition_axis, None, self.recurrent_partition_axis, None, None),
        )

    def _inner_sharding(self, i: int):
        return (
            self.conv_sharding,
            self.spec_conv_sharding,
            self.spec_recurrent_sharding,
            NamedSharding(self.mesh, P(self.data_partition_axis)),
        )[i]

    def commit_speculative_state(self, recurrent_indices, accept_lens) -> None:
        """After target verify: running state of each request := its verify state after
        ``accept_lens`` tokens (bonus included, so step ``accept_lens - 1``). Rows with
        slot 0 (padding) or ``accept_lens <= 0`` are left alone."""
        assert self.spec_num_draft_tokens > 0, "pool has no speculative verify scratch"
        new_recurrent, new_conv = _commit_speculative_state(
            self.recurrent_buffers,
            self.conv_buffers,
            recurrent_indices,
            accept_lens,
            mesh=self.mesh,
            data_axis=self.data_partition_axis,
            recurrent_axis=self.recurrent_partition_axis,
            conv_axis=self.conv_partition_axis,
            conv_tiled=self.conv_tiled,
            lazy_commit=self.lazy_commit,
        )
        self.recurrent_buffers = list(new_recurrent)
        self.conv_buffers = [list(inner) for inner in new_conv]

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        return self.recurrent_buffers[idx], self.conv_buffers[idx]

    def replace_buffer(self, buffers) -> None:
        new_recurrent, new_conv = buffers

        assert len(new_recurrent) == self.num_linear_recurrent_layers
        assert len(new_conv) == self.num_linear_recurrent_layers

        # tp_size==1 sharding fix: see MHATokenToKVPool.replace_buffer
        tp_degenerate = self.mesh.shape.get("tensor", 1) == 1
        for layer in range(self.num_linear_recurrent_layers):
            buf = new_recurrent[layer]
            if tp_degenerate:
                buf = jax.device_put(buf, self.recurrent_sharding)
            self.recurrent_buffers[layer] = buf

            assert len(new_conv[layer]) == len(self.conv_buffers[layer])
            for i in range(len(new_conv[layer])):
                cbuf = new_conv[layer][i]
                if tp_degenerate:
                    cbuf = jax.device_put(cbuf, self._inner_sharding(i))
                self.conv_buffers[layer][i] = cbuf

    def clear(self) -> None:
        for layer in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[layer] = jnp.zeros_like(self.recurrent_buffers[layer])
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = jnp.zeros_like(self.conv_buffers[layer][inner])

    def copy_slots(self, src_indices, dst_indices):
        """Clone src->dst slots across all layers; rows with src==0 keep dst.
        Indices are per-DP-rank local; returns new buffers for the donated pool."""
        mesh = self.mesh
        data_axis = self.data_partition_axis

        def _temporal(buf, src, dst):
            # Donated-buffer aliasing barriers: without them the scatter races the
            # gather under multi-host SPMD -> NaN. Value-preserving; do not remove.
            buf = jax.lax.optimization_barrier(buf)
            val = jnp.where((src == 0).reshape(-1, 1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        def _conv(buf, src, dst):
            buf = jax.lax.optimization_barrier(buf)  # see _temporal
            val = jnp.where((src == 0).reshape((-1,) + (1,) * (buf.ndim - 1)), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        copy_temporal = jax.shard_map(
            _temporal,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.recurrent_partition_axis, None, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.recurrent_partition_axis, None, None),
            check_vma=False,
        )
        conv_spec = self.conv_sharding.spec
        copy_conv = jax.shard_map(
            _conv,
            mesh=mesh,
            in_specs=(conv_spec, P(data_axis), P(data_axis)),
            out_specs=conv_spec,
            check_vma=False,
        )

        new_recurrent = [
            copy_temporal(buf, src_indices, dst_indices) for buf in self.recurrent_buffers
        ]
        # Only the running conv state is copied; speculative verify scratch is per-forward.
        new_conv = [
            [copy_conv(inner[0], src_indices, dst_indices)] + list(inner[1:])
            for inner in self.conv_buffers
        ]
        return new_recurrent, new_conv

    # --- pytree ---
    def tree_flatten(self):
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.size,
            self.dp_size,
            self.total_slots,
            self.num_heads,
            self.head_dim,
            self.num_k_heads,
            self.head_k_dim,
            self.conv_kernel_size,
            self.temporal_dtype,
            self.conv_dtype,
            self.mesh,
            self.recurrent_partition_axis,
            self.conv_partition_axis,
            self.data_partition_axis,
            self.recurrent_sharding,
            self.conv_sharding,
            self.spec_num_draft_tokens,
            self.conv_tiled,
            self.lazy_commit,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            linear_recurrent_layer_ids_tup,
            size,
            dp_size,
            total_slots,
            num_heads,
            head_dim,
            num_k_heads,
            head_k_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
            mesh,
            recurrent_partition_axis,
            conv_partition_axis,
            data_partition_axis,
            recurrent_sharding,
            conv_sharding,
            spec_num_draft_tokens,
            conv_tiled,
            lazy_commit,
        ) = aux_data
        obj = cls.__new__(cls)
        obj.spec_num_draft_tokens = spec_num_draft_tokens
        obj.conv_tiled = conv_tiled
        obj.lazy_commit = lazy_commit
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.size = size
        obj.dp_size = dp_size
        obj.slots_per_rank = size // dp_size
        obj.total_slots = total_slots
        obj.num_heads = num_heads
        obj.head_dim = head_dim
        obj.num_k_heads = num_k_heads
        obj.head_k_dim = head_k_dim
        obj.conv_kernel_size = conv_kernel_size
        obj.temporal_dtype = temporal_dtype
        obj.conv_dtype = conv_dtype
        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        obj.proj_size = proj_v + 2 * proj_k
        obj.mesh = mesh
        obj.recurrent_partition_axis = recurrent_partition_axis
        obj.conv_partition_axis = conv_partition_axis
        obj.data_partition_axis = data_partition_axis
        obj.recurrent_sharding = recurrent_sharding
        obj.conv_sharding = conv_sharding
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        return obj


@partial(
    jax.jit,
    static_argnames=("mesh", "data_axis", "recurrent_axis", "conv_axis", "conv_tiled", "lazy_commit"),
    donate_argnums=(0, 1),
)
def _commit_speculative_state(
    recurrent_buffers,
    conv_buffers,
    recurrent_indices,
    accept_lens,
    *,
    mesh,
    data_axis,
    recurrent_axis,
    conv_axis,
    conv_tiled=False,
    lazy_commit=False,
):
    """Per layer and batch row b: ``ssm[idx[b]] = spec_ssm[b, a[b]-1]`` (same for conv).

    The verify scratch is indexed by batch row (see ``KDAAttnBackend._forward_target_verify``).
    """

    if lazy_commit:
        # Slot-indexed scratch: record the accepted step; no state is copied.
        def _record(steps, idx, acc):
            dst = jnp.where((idx == 0) | (acc <= 0), 0, idx)
            return steps.at[dst].set(jnp.where(dst == 0, -1, acc - 1).astype(steps.dtype))

        record = jax.shard_map(
            _record, mesh=mesh, in_specs=(P(data_axis),) * 3, out_specs=P(data_axis),
            check_vma=False,
        )
        new_conv = [
            list(inner[:3]) + [record(inner[3], recurrent_indices, accept_lens)]
            for inner in conv_buffers
        ]
        return list(recurrent_buffers), new_conv

    def _select(buf, spec, idx, acc):
        buf = jax.lax.optimization_barrier(buf)  # donated-buffer aliasing guard, see copy_slots
        steps = spec.shape[1]
        step = jnp.clip(acc - 1, 0, steps - 1)
        val = spec[jnp.arange(idx.shape[0]), step]
        # Rows to leave alone (padding slot 0, accept <= 0) write their (finite) verify state
        # into the reserved dummy slot 0 instead of re-writing their own row: no gather of
        # buf[idx] and no select.
        dst = jnp.where((idx == 0) | (acc <= 0), 0, idx)
        return jax.lax.optimization_barrier(buf.at[dst].set(val))

    commit_ssm = jax.shard_map(
        _select,
        mesh=mesh,
        in_specs=(
            P(data_axis, recurrent_axis, None, None),
            P(data_axis, None, recurrent_axis, None, None),
            P(data_axis),
            P(data_axis),
        ),
        out_specs=P(data_axis, recurrent_axis, None, None),
        check_vma=False,
    )
    conv_spec = P(data_axis, None, None, conv_axis, None) if conv_tiled else P(data_axis, conv_axis, None)
    spec_conv_spec = (
        P(data_axis, None, None, None, conv_axis, None) if conv_tiled
        else P(data_axis, None, conv_axis, None)
    )
    commit_conv = jax.shard_map(
        _select,
        mesh=mesh,
        in_specs=(conv_spec, spec_conv_spec, P(data_axis), P(data_axis)),
        out_specs=conv_spec,
        check_vma=False,
    )
    new_recurrent = []
    new_conv = []
    for rec, inner in zip(recurrent_buffers, conv_buffers):
        conv, spec_conv, spec_ssm = inner
        new_recurrent.append(commit_ssm(rec, spec_ssm, recurrent_indices, accept_lens))
        new_conv.append([commit_conv(conv, spec_conv, recurrent_indices, accept_lens), spec_conv, spec_ssm])
    return new_recurrent, new_conv
