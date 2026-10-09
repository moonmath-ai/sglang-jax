"""Host-side MLA speculative metadata: TARGET_VERIFY / DRAFT_EXTEND as T causal queries per
request and per-step draft decode metadata, all over one fixed-stride request page table.

    JAX_PLATFORMS=cpu python -m pytest python/sgl_jax/test/speculative/test_mla_spec_metadata.py
"""

import unittest
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

PS = 4  # page size


def _backend(mesh):
    return MLAAttentionBackend(
        num_attn_heads=4,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=0,
        v_head_dim=16,
        page_size=PS,
        mesh=mesh,
        vmem_limit_bytes=1 << 20,
    )


def _pack_cache_loc(pages_per_req, allocate_lens, total):
    """Mirror EagleDraftWorker.padding_for_decode: page-start token locs, packed per
    request over its page-aligned allocate_len."""
    cache_loc = np.zeros(total, np.int32)
    base = 0
    for pages, alloc in zip(pages_per_req, allocate_lens):
        aligned = -(-alloc // PS) * PS
        cache_loc[base : base + aligned : PS] = np.asarray(pages) * PS
        base += aligned
    return cache_loc


class TestMLASpecMetadata(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
        self.backend = _backend(self.mesh)
        # 2 live requests + 1 padding slot. Request 0 holds more pages (5) than the batch's
        # average page capacity (48 / PS / 3 = 4): its page table must not be truncated.
        self.pages = [[7, 3, 9, 2, 8], [5]]
        self.allocate_lens = np.array([20, 4])
        self.cache_loc = _pack_cache_loc(self.pages, self.allocate_lens, 48)

    def _check(self, md, kv_lens, T, stride):
        n_valid = int(np.sum(np.asarray(kv_lens) > 0))
        np.testing.assert_array_equal(np.asarray(md.seq_lens), kv_lens)
        np.testing.assert_array_equal(np.asarray(md.cu_q_lens), np.arange(4) * T)
        np.testing.assert_array_equal(np.asarray(md.cu_kv_lens), np.arange(4) * stride * PS)
        dist = [n_valid] * 3 if T == 1 else [0, 0, n_valid]
        np.testing.assert_array_equal(np.asarray(md.distribution), dist)
        pi = np.asarray(md.page_indices).reshape(3, stride)
        for r, pages in enumerate(self.pages):
            np.testing.assert_array_equal(pi[r, : len(pages)], pages)

    def _verify_batch(self, T):
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            dp_size=1,
            seq_lens=np.array([15, 1, 0]),  # committed KV lengths (after prepare_for_verify)
            cache_loc=self.cache_loc,
            spec_info_padded=SimpleNamespace(draft_token_num=T, allocate_lens=self.allocate_lens),
        )

    def test_target_verify(self):
        T = 4
        md = self.backend.get_eagle_forward_metadata(self._verify_batch(T))
        # No max_pages_per_seq: the stride is the batch's whole page capacity.
        self._check(md, [19, 5, 0], T, stride=48 // PS)

    def test_target_verify_max_pages_stride(self):
        T = 4
        self.backend.max_pages_per_seq = 6
        md = self.backend.get_eagle_forward_metadata(self._verify_batch(T))
        self._check(md, [19, 5, 0], T, stride=6)

    def test_draft_extend(self):
        T = 4
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            dp_size=1,
            seq_lens=np.array([15 + T - 1, 1 + T - 1, 0]),
            input_ids=np.zeros(3 * T, np.int32),
            cache_loc=self.cache_loc,
            spec_info_padded=SimpleNamespace(allocate_lens=self.allocate_lens),
        )
        md = self.backend.get_eagle_forward_metadata(batch)
        self._check(md, [19, 5, 0], T, stride=48 // PS)

    def test_multi_step(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            dp_size=1,
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            seq_lens=np.array([16, 2, 0]),
            cache_loc=self.cache_loc,
            spec_info_padded=SimpleNamespace(allocate_lens=np.array([20, 4, 0])),
        )
        mds = self.backend.get_eagle_multi_step_metadata(batch)
        self.assertEqual(len(mds), 3)
        for i, md in enumerate(mds):
            self._check(md, [16 + i, 2 + i, 0], 1, stride=48 // PS)


if __name__ == "__main__":
    unittest.main()
