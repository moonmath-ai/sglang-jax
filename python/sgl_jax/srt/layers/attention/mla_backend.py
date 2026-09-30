"""Attention backend for the absorbed MLA path (MLA v2 Pallas kernel).

This backend wraps `mla_ragged_paged_attention` with a 4D paged latent KV cache
(see `MLATokenToKVPool`). It follows the same `(q, k, v, layer, forward_batch,
pool, **kwargs)` contract as the MHA FlashAttention backend so that
`RadixAttention` remains the unified entry point. Callers pass the absorbed
latent Q as `q`, the latent `c_kv` as both `k` and `v`, and the rope parts as
the `q_rope` / `k_rope` kwargs; the backend reassembles the 4-tuple
`(ql_nope, q_pe, new_kv_c, new_k_pe)` the Pallas kernel consumes. The output
is the latent attention `o_latent: [T, n_h, kv_lora_rank]`; the caller
projects it through `W_UV → W_O` (see `docs/design/MLA.md` §3.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.mla.v2.kernel import cdiv, mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class MLAAttentionMetadata:
    """Per-forward metadata for the MLA v2 kernel.

    Drops `custom_mask` / `swa_page_indices` / `attention_sink` (RFC §3.8). The
    `cu_kv_lens` field carries the page-aligned KV cumulative lengths used by
    the kernel to locate each sequence's pages in the ragged `page_indices`
    layout (matches RPA v3 addressing).
    """

    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None
    # Speculative chains only (TARGET_VERIFY / DRAFT_EXTEND): the same batch in request
    # layout -- one sequence of T causal queries per request, fixed-stride page table --
    # so per-query work that only reads the cache (the DSA indexer top-k) scans each
    # request's context once instead of once per virtual sequence.
    spec_req_seq_lens: jax.Array = None
    spec_req_page_indices: jax.Array = None
    spec_req_cu_q_lens: jax.Array = None
    spec_req_distribution: jax.Array = None

    def tree_flatten(self):
        children = (
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
            self.spec_req_seq_lens,
            self.spec_req_page_indices,
            self.spec_req_cu_q_lens,
            self.spec_req_distribution,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (
            obj.cu_q_lens,
            obj.cu_kv_lens,
            obj.page_indices,
            obj.seq_lens,
            obj.distribution,
            obj.spec_req_seq_lens,
            obj.spec_req_page_indices,
            obj.spec_req_cu_q_lens,
            obj.spec_req_distribution,
        ) = children
        return obj


@dataclass
class MLAAttentionBackend(AttentionBackend):
    """Absorbed-MLA attention backend backed by the v2 Pallas kernel."""

    def __init__(
        self,
        num_attn_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        page_size: int = 1,
        mesh: jax.sharding.Mesh = None,
        attention_data_partition_axis: str = "data",
        vmem_limit_bytes: int | None = None,
        # Default block params to None so the kernel's auto-tuned-table
        # lookup path triggers (see kernel.py mla_ragged_paged_attention
        # entry). Callers (e.g. accuracy tests) can still pass explicit
        # tuples to bypass lookup.
        num_kv_pages_per_block: tuple[int, int, int] | None = None,
        num_queries_per_block: tuple[int, int, int] | None = None,
        # decode_batch_size: kernel-internal microbatch for the BATCHED_DECODE
        # branch. The v2 kernel runs `floor(num_decode_seqs / decode_batch_size)
        # * decode_batch_size` requests through a batched path (q_len=1, q-tile
        # of size `decode_batch_size`) to keep the MXU busy on MLA's MQA-on-
        # latent decode (single q row per seq otherwise underutilizes the q
        # axis); the remainder falls back to per-seq decode.
        # When num_kv_pages_per_block is None, the kernel's lookup path also
        # overrides this with the tuned value if the "decode" entry hits.
        decode_batch_size: int = 4,
    ):
        assert page_size > 1, (
            "MLA attention backend does not support page_size=1. "
            "The MLA v2 kernel packs KV slots and infers the effective page size "
            "from cache_kv.shape[1] * kv_packing, which makes page_size=1 "
            "inconsistent with allocator and metadata page semantics."
        )
        self.num_heads = num_attn_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.mesh = mesh
        self.attention_data_partition_axis = attention_data_partition_axis
        if vmem_limit_bytes is None:
            if jax.default_backend() == "tpu":
                from jax.experimental.pallas import tpu as pltpu

                vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
            else:
                # CPU (Pallas interpret) runs have no VMEM to size against.
                vmem_limit_bytes = 64 * 1024 * 1024
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_kv_pages_per_block = num_kv_pages_per_block
        self.num_queries_per_block = num_queries_per_block
        self.decode_batch_size = decode_batch_size

        self.forward_metadata = nnx.data(MLAAttentionMetadata())

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Build per-batch metadata, DP-aware.

        Mirrors `FlashAttention.get_forward_metadata`: reshapes all per-request
        arrays to `(dp_size, per_dp_bs_size)` and computes cumsums per DP rank
        so each rank's metadata is independent.
        """
        metadata = MLAAttentionMetadata()

        if batch.dp_size <= 0:
            raise ValueError(f"Invalid dp_size: {batch.dp_size}")
        if batch.per_dp_bs_size <= 0:
            raise ValueError(f"Invalid per_dp_bs_size: {batch.per_dp_bs_size}")
        if batch.per_dp_bs_size * batch.dp_size != len(batch.seq_lens):
            raise ValueError(
                "Inconsistent DP batch metadata: expected per_dp_bs_size * dp_size == len(seq_lens), "
                f"got {batch.per_dp_bs_size} * {batch.dp_size} != {len(batch.seq_lens)}"
            )
        if len(batch.cache_loc) % batch.dp_size != 0:
            raise ValueError(
                "Inconsistent cache_loc layout for DP sharding: "
                f"len(cache_loc)={len(batch.cache_loc)} is not divisible by dp_size={batch.dp_size}"
            )

        total_loc_len = len(batch.cache_loc)
        per_dp_loc_len = total_loc_len // batch.dp_size

        cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
        strided_2d = cache_loc_2d[:, :: self.page_size]
        page_indices = (strided_2d // self.page_size).ravel()

        if batch.forward_mode == ForwardMode.EXTEND:
            ext_2d = batch.extend_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
            cu_q_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
            cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
            cu_q_lens = cu_q_2d.ravel()
        elif batch.forward_mode == ForwardMode.DECODE:
            single_cu = np.arange(batch.per_dp_bs_size + 1, dtype=np.int32)
            cu_q_lens = np.tile(single_cu, batch.dp_size)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = batch.seq_lens
        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size

        aligned_2d = aligned_seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        cu_kv_2d = np.zeros((batch.dp_size, batch.per_dp_bs_size + 1), dtype=np.int32)
        cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)
        cu_kv_lens = cu_kv_2d.ravel()

        seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, batch.per_dp_bs_size)
        local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.repeat(local_num_seqs, 3)
        elif batch.forward_mode == ForwardMode.EXTEND:
            distribution = np.column_stack(
                [np.zeros_like(local_num_seqs), np.zeros_like(local_num_seqs), local_num_seqs]
            ).ravel()
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P(self.attention_data_partition_axis))),
        )
        return metadata

    # ------------------------------------------------------------------
    # Speculative decoding (EAGLE / NEXTN, topk=1 chain)
    # ------------------------------------------------------------------
    #
    # Every spec forward is expressed as a *decode* batch so the one-query-per-sequence
    # kernels (MLA decode, DSA indexer top-k + page-level sparse MLA) apply unchanged:
    #
    #   * TARGET_VERIFY / DRAFT_EXTEND: T tokens per request at KV slots S..S+T-1 become T
    #     "virtual" sequences sharing the request's page table, virtual seq j having
    #     kv_len = S + j + 1 (causal chain, the new token is the last KV slot).
    #   * draft decode step i: one token per request with kv_len = seq_lens + i.
    #
    # ``S`` is the committed KV length: ``seq_lens`` after ``prepare_for_verify`` for
    # TARGET_VERIFY, ``seq_lens - (T - 1)`` after ``prepare_for_extend_after_verify`` for
    # DRAFT_EXTEND. Page tables come from ``padding_for_decode``'s ``cache_loc``, which packs
    # each valid request's page-start token locations over its page-aligned
    # ``allocate_len`` region. dp_size == 1 only.

    def _spec_request_pages(self, batch: ModelWorkerBatch):
        """Per padded request slot: its physical page ids (None for padding slots)."""
        if batch.dp_size != 1:
            raise NotImplementedError("MLA speculative metadata supports dp_size == 1 only")
        seq_lens = np.asarray(batch.seq_lens)
        valid = seq_lens > 0
        allocate_lens = np.asarray(batch.spec_info_padded.allocate_lens)
        if len(allocate_lens) == len(seq_lens):
            allocate_lens = allocate_lens[valid]
        n_valid = int(valid.sum())
        assert len(allocate_lens) >= n_valid, (len(allocate_lens), n_valid)
        cache_loc = np.asarray(batch.cache_loc)
        pages: list[np.ndarray | None] = [None] * len(seq_lens)
        base = 0
        for slot, alloc in zip(np.nonzero(valid)[0], allocate_lens[:n_valid]):
            aligned = int(cdiv(int(alloc), self.page_size)) * self.page_size
            pages[slot] = cache_loc[base : base + aligned : self.page_size] // self.page_size
            base += aligned
        return pages, len(cache_loc) // self.page_size

    def _virtual_decode_arrays(
        self, slot_pages, kv_lens: np.ndarray, virtual_slot: np.ndarray, page_capacity: int
    ) -> tuple[np.ndarray, ...]:
        """Host arrays ``(cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution)`` of a
        decode batch where virtual seq v attends ``kv_lens[v]`` tokens of request
        ``virtual_slot[v]``'s pages. Valid virtual seqs must form a prefix."""
        num_seqs = len(kv_lens)
        n_pages = cdiv(kv_lens, self.page_size).astype(np.int64)
        assert int(n_pages.sum()) <= page_capacity, (int(n_pages.sum()), page_capacity)
        page_indices = np.zeros(page_capacity, dtype=np.int32)
        off = 0
        for v in range(num_seqs):
            n = int(n_pages[v])
            if n:
                page_indices[off : off + n] = slot_pages[virtual_slot[v]][:n]
                off += n
        cu_kv_lens = np.zeros(num_seqs + 1, dtype=np.int32)
        cu_kv_lens[1:] = np.cumsum(n_pages * self.page_size)
        cu_q_lens = np.arange(num_seqs + 1, dtype=np.int32)
        n_valid = int(np.sum(kv_lens > 0))
        assert np.all(kv_lens[:n_valid] > 0) and np.all(kv_lens[n_valid:] == 0)
        distribution = np.array([n_valid, n_valid, n_valid], dtype=np.int32)
        return cu_q_lens, cu_kv_lens, page_indices, kv_lens.astype(np.int32), distribution

    def metadata_from_arrays(self, arrays, *, upload: bool = True) -> MLAAttentionMetadata:
        """Wrap ``_virtual_decode_arrays`` output (5 arrays), optionally followed by the
        4 ``spec_req_*`` request-layout arrays (host, or already on device)."""
        if upload:
            arrays = device_array(
                tuple(arrays),
                sharding=(NamedSharding(self.mesh, P(self.attention_data_partition_axis))),
            )
        return MLAAttentionMetadata(*arrays)

    def _spec_request_layout_arrays(self, slot_pages, committed, valid, T, pages_per_seq):
        """Request-layout view of a T-token chain: seq_len = S + T, T causal queries."""
        bs = len(committed)
        seq = np.where(valid, committed + T, 0).astype(np.int32)
        pi = np.zeros((bs, pages_per_seq), dtype=np.int32)
        for r in np.nonzero(valid)[0]:
            n = int(cdiv(int(seq[r]), self.page_size))
            pi[r, :n] = slot_pages[r][:n]
        cu_q = (np.arange(bs + 1) * T).astype(np.int32)
        n_valid = int(valid.sum())
        dist = np.array([0, 0, n_valid], dtype=np.int32)  # all "mixed": multi-query rows
        return seq, pi.reshape(-1), cu_q, dist

    def eagle_forward_metadata_arrays(self, batch: ModelWorkerBatch):
        """Host arrays for TARGET_VERIFY / DRAFT_EXTEND (see get_eagle_forward_metadata)."""
        seq_lens = np.asarray(batch.seq_lens, dtype=np.int64)
        bs = len(seq_lens)
        if batch.forward_mode == ForwardMode.TARGET_VERIFY:
            T = int(batch.spec_info_padded.draft_token_num)
            committed = seq_lens
        elif batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            T = int(np.asarray(batch.input_ids).shape[0]) // bs
            committed = seq_lens - (T - 1)
        else:
            raise ValueError(f"Invalid speculative forward mode: {batch.forward_mode}")
        slot_pages, total_pages = self._spec_request_pages(batch)
        valid = seq_lens > 0
        j = np.arange(T, dtype=np.int64)
        kv_lens = np.where(valid[:, None], committed[:, None] + j[None, :] + 1, 0).reshape(-1)
        virtual_slot = np.repeat(np.arange(bs), T)
        virtual = self._virtual_decode_arrays(slot_pages, kv_lens, virtual_slot, T * total_pages)
        # Same fixed stride as the virtual layout's pages_per_seq (= len(pi) // num_seqs).
        req = self._spec_request_layout_arrays(
            slot_pages, committed, valid, T, (T * total_pages) // (T * bs)
        )
        return virtual + req

    def eagle_multi_step_metadata_arrays(self, batch: ModelWorkerBatch):
        """Host arrays per draft decode step i (kv_len = seq_lens + i)."""
        assert batch.forward_mode == ForwardMode.DECODE
        assert batch.speculative_eagle_topk == 1, "MLA speculative decoding requires topk == 1"
        seq_lens = np.asarray(batch.seq_lens, dtype=np.int64)
        slot_pages, total_pages = self._spec_request_pages(batch)
        valid = seq_lens > 0
        virtual_slot = np.arange(len(seq_lens))
        return [
            self._virtual_decode_arrays(
                slot_pages, np.where(valid, seq_lens + i, 0), virtual_slot, total_pages
            )
            for i in range(batch.speculative_num_steps)
        ]

    def get_eagle_forward_metadata(self, batch: ModelWorkerBatch, **kwargs):
        """Metadata for draft prefill (EXTEND), TARGET_VERIFY and DRAFT_EXTEND."""
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.get_forward_metadata(batch)
        return self.metadata_from_arrays(self.eagle_forward_metadata_arrays(batch))

    def get_eagle_multi_step_metadata(self, batch: ModelWorkerBatch):
        """One decode metadata per draft step i: kv_len = seq_lens + i."""
        return [self.metadata_from_arrays(a) for a in self.eagle_multi_step_metadata_arrays(batch)]

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_attn_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "page_size": self.page_size,
            "mesh": self.mesh,
            "attention_data_partition_axis": self.attention_data_partition_axis,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "num_kv_pages_per_block": self.num_kv_pages_per_block,
            "num_queries_per_block": self.num_queries_per_block,
            "decode_batch_size": self.decode_batch_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            num_attn_heads=aux_data["num_attn_heads"],
            kv_lora_rank=aux_data["kv_lora_rank"],
            qk_nope_head_dim=aux_data["qk_nope_head_dim"],
            qk_rope_head_dim=aux_data["qk_rope_head_dim"],
            v_head_dim=aux_data["v_head_dim"],
            page_size=aux_data["page_size"],
            mesh=aux_data.get("mesh"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            vmem_limit_bytes=aux_data["vmem_limit_bytes"],
            num_kv_pages_per_block=aux_data["num_kv_pages_per_block"],
            num_queries_per_block=aux_data["num_queries_per_block"],
            decode_batch_size=aux_data["decode_batch_size"],
        )
        obj.forward_metadata = children[0]
        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        """Absorbed-MLA forward, called through ``RadixAttention``.

        Signature mirrors the other attention backends so ``RadixAttention``
        can serve as the unified entry point. The MLA-specific tensors land in
        standard slots:

            q:         [T, n_h, kv_lora_rank]            — absorbed latent query (``ql_nope``)
            k:         [T, 1, kv_lora_rank]              — latent KV (c_kv); v is the same tensor
            v:         [T, 1, kv_lora_rank]              — same object as ``k`` (MLA is MQA on the latent)
            q_rope=…:  [T, n_h, qk_rope_head_dim]        — ``q_pe``
            k_rope=…:  [T, 1, qk_rope_head_dim]          — ``k_pe`` (shared across heads)

        Internally reassembles the 4-tuple ``(ql_nope, q_pe, new_kv_c, new_k_pe)``
        the v2 Pallas kernel consumes. Returns ``(o_latent, updated_cache)``
        with ``o_latent: [T, n_h, kv_lora_rank]`` — the caller is responsible
        for projecting via ``W_UV → W_O``.

        The kernel handles ``qk_rope_head_dim=64 → 128`` and other
        ``align_to(*, 128)`` padding internally, so callers pass logical dims.
        """
        # v is the same latent tensor as k in MLA-MQA — we only need one.
        del v
        q_rope = kwargs.get("q_rope")
        k_rope = kwargs.get("k_rope")
        if q_rope is None or k_rope is None:
            raise ValueError(
                "MLAAttentionBackend requires q_rope/k_rope kwargs (q_pe/k_pe) "
                "alongside the non-rope q/k tensors."
            )

        # Strip the (single) KV-head axis from the latent K/K-rope tensors.
        # Squeezing a length-1 axis preserves any caller-provided sharding on
        # the remaining dims (replicated stays replicated).
        new_kv_c = k if k.ndim == 2 else jnp.squeeze(k, axis=1)
        new_k_pe = k_rope if k_rope.ndim == 2 else jnp.squeeze(k_rope, axis=1)
        dpa = self.attention_data_partition_axis
        new_k_pe = jax.sharding.reshard(new_k_pe, P(dpa, None))
        ql_nope = q
        q_pe = q_rope

        cache = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        sm_scale = (
            (1.0 / jnp.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )
        sliding_window = layer.sliding_window_size if layer is not None else None
        soft_cap = layer.logit_cap if layer is not None else None

        in_specs = (
            P(dpa, "tensor", None),  # ql_nope    [T, n_h/tp, lkv]
            P(dpa, "tensor", None),  # q_pe       [T, n_h/tp, r]
            P(dpa, None),  # new_kv_c   [T, lkv]  (single latent, no head axis)
            P(dpa, None),  # new_k_pe   [T, r]    (single latent)
            P(dpa, None, None, None),  # cache (page axis sharded by data)
            P(dpa),  # seq_lens
            P(dpa),  # page_indices
            P(dpa),  # cu_q_lens
            P(dpa),  # cu_kv_lens
            P(dpa),  # distribution
        )
        out_specs = (
            P(dpa, "tensor", None),  # o_latent       [T, n_h/tp, lkv]
            P(dpa, None, None, None),  # updated cache  4D
        )

        def _run(
            ql_nope_,
            q_pe_,
            new_kv_c_,
            new_k_pe_,
            cache_,
            seq_lens_,
            page_indices_,
            cu_q_lens_,
            cu_kv_lens_,
            distribution_,
        ):
            return mla_ragged_paged_attention(
                ql_nope_,
                q_pe_,
                new_kv_c_,
                new_k_pe_,
                cache_,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                num_kv_pages_per_block=self.num_kv_pages_per_block,
                num_queries_per_block=self.num_queries_per_block,
                decode_batch_size=self.decode_batch_size,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        o_latent, updated_cache = jax.shard_map(
            _run,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache,
            self.forward_metadata.seq_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
        )

        return o_latent, updated_cache

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # Same scalar-prefetch budget heuristic as FlashAttention.get_max_running_reqests:
        # caps the number of concurrent requests by the per-request page slot count
        # so metadata arrays fit in TPU scalar-prefetch memory.
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res

    def get_eagle_base_metadata(self, batch, **kwargs):
        """Metadata for the fused speculative JITs that rebuild dynamic fields on device.

        The FlashAttention backend uploads only page ids here; the MLA/DSA verify and
        draft-extend helpers (``_make_target_verify_metadata`` /
        ``_make_draft_extend_metadata``) rebuild seq_lens, cu_q_lens, cu_kv_lens and the
        page table from ``page_indices`` themselves, so the full host metadata is a
        valid (if slightly larger) base. Keep one construction path until MLA grows a
        page-only upload.
        """
        return self.get_eagle_forward_metadata(batch, **kwargs)

    def get_eagle_forward_metadata(self, batch, **kwargs):
        # We don't support custom_mask in MLA yet, but for NEXTN it's just sequential draft tokens.
        import numpy as np

        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        metadata = MLAAttentionMetadata()

        dp_size = batch.dp_size
        per_dp_bs = batch.per_dp_bs_size if dp_size > 1 else len(batch.seq_lens)

        if batch.forward_mode.is_target_verify():
            padded_batch_size = len(batch.seq_lens)
            extend_seq_lens = np.zeros(padded_batch_size, dtype=np.int32)
            extend_seq_lens[batch.logits_indices_selector] = batch.spec_info_padded.draft_token_num
        else:
            extend_seq_lens = batch.extend_seq_lens

        ext_2d = extend_seq_lens.reshape(dp_size, per_dp_bs)
        cu_q_2d = np.zeros((dp_size, per_dp_bs + 1), dtype=np.int32)
        cu_q_2d[:, 1:] = np.cumsum(ext_2d, axis=1)
        metadata.cu_q_lens = cu_q_2d.ravel()

        total_loc_len = len(batch.cache_loc)
        per_dp_loc_len = total_loc_len // batch.dp_size
        cache_loc_2d = batch.cache_loc.reshape(batch.dp_size, per_dp_loc_len)
        strided_2d = cache_loc_2d[:, :: self.page_size]
        metadata.page_indices = (strided_2d // self.page_size).ravel()

        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size
        aligned_2d = aligned_seq_lens.reshape(batch.dp_size, per_dp_bs)
        cu_kv_2d = np.zeros((batch.dp_size, per_dp_bs + 1), dtype=np.int32)
        cu_kv_2d[:, 1:] = np.cumsum(aligned_2d, axis=1)
        metadata.cu_kv_lens = cu_kv_2d.ravel()

        seq_lens_2d = batch.seq_lens.reshape(batch.dp_size, per_dp_bs)
        metadata.seq_lens = seq_lens_2d.ravel()

        local_num_seqs = np.sum(seq_lens_2d > 0, axis=1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.repeat(local_num_seqs, 3)
        else:
            distribution = np.column_stack(
                [np.zeros_like(local_num_seqs), np.zeros_like(local_num_seqs), local_num_seqs]
            ).ravel()
        metadata.distribution = distribution

        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.utils.jax_utils import device_array

        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (
                metadata.cu_q_lens,
                metadata.cu_kv_lens,
                metadata.page_indices,
                metadata.seq_lens,
                metadata.distribution,
            ),
            sharding=(NamedSharding(self.mesh, P(self.attention_data_partition_axis))),
        )

        return metadata
