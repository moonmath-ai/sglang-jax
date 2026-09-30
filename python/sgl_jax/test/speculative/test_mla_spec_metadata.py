"""Host-side MLA speculative metadata: TARGET_VERIFY / DRAFT_EXTEND as one virtual decode
sequence per chain token, and per-step draft decode metadata.

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
        # 2 live requests + 1 padding slot.
        self.pages = [[7, 3, 9], [5, 11]]
        self.allocate_lens = np.array([12, 8])
        # Like padding_for_decode: bs_bucket * max_req_len tokens (3 slots x 4 pages).
        self.cache_loc = _pack_cache_loc(self.pages, self.allocate_lens, 48)

    def _check_virtual(self, md, kv_lens, owner):
        """Lengths per virtual seq; page regions cover the owner's whole allocation."""
        seq_lens = np.asarray(md.seq_lens)
        np.testing.assert_array_equal(seq_lens, kv_lens)
        cu_kv = np.asarray(md.cu_kv_lens)
        pi = np.asarray(md.page_indices)
        np.testing.assert_array_equal(np.asarray(md.cu_q_lens), np.arange(len(kv_lens) + 1))
        n_valid = int(np.sum(np.asarray(kv_lens) > 0))
        np.testing.assert_array_equal(np.asarray(md.distribution), [n_valid] * 3)
        for v, (kv, r) in enumerate(zip(kv_lens, owner)):
            if not kv:
                self.assertEqual(cu_kv[v + 1] - cu_kv[v], 0)
                continue
            region = self.pages[r]
            self.assertEqual(cu_kv[v + 1] - cu_kv[v], len(region) * PS)
            self.assertLessEqual(kv, len(region) * PS)
            np.testing.assert_array_equal(pi[cu_kv[v] // PS : cu_kv[v] // PS + len(region)], region)

    def test_target_verify(self):
        T = 4
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            dp_size=1,
            seq_lens=np.array([6, 3, 0]),  # committed KV lengths (after prepare_for_verify)
            cache_loc=self.cache_loc,
            spec_info_padded=SimpleNamespace(draft_token_num=T, allocate_lens=self.allocate_lens),
        )
        md = self.backend.get_eagle_forward_metadata(batch)
        kv = [7, 8, 9, 10, 4, 5, 6, 7, 0, 0, 0, 0]
        self._check_virtual(md, kv, [0] * 4 + [1] * 4 + [0] * 4)
        self.assertEqual(np.asarray(md.page_indices).shape[0], T * 48 // PS)
        self._check_request_layout(md, committed=[6, 3, 0], T=T)

    def test_draft_extend(self):
        T = 4
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            dp_size=1,
            seq_lens=np.array([6 + T - 1, 3 + T - 1, 0]),
            input_ids=np.zeros(3 * T, np.int32),
            cache_loc=self.cache_loc,
            spec_info_padded=SimpleNamespace(allocate_lens=self.allocate_lens),
        )
        md = self.backend.get_eagle_forward_metadata(batch)
        self._check_virtual(md, [7, 8, 9, 10, 4, 5, 6, 7, 0, 0, 0, 0], [0] * 4 + [1] * 4 + [0] * 4)
        self._check_request_layout(md, committed=[6, 3, 0], T=T)

    def _check_request_layout(self, md, committed, T):
        seq = np.asarray(md.spec_req_seq_lens)
        np.testing.assert_array_equal(seq, [c + T if c else 0 for c in committed])
        np.testing.assert_array_equal(np.asarray(md.spec_req_cu_q_lens), np.arange(4) * T)
        np.testing.assert_array_equal(np.asarray(md.spec_req_distribution), [0, 0, 2])
        pps = 48 // PS // 3
        pi = np.asarray(md.spec_req_page_indices).reshape(3, pps)
        for r, n in enumerate(-(-seq // PS)):
            if n:
                np.testing.assert_array_equal(pi[r, :n], self.pages[r][:n])

    def test_multi_step(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            dp_size=1,
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            seq_lens=np.array([7, 4, 0]),
            cache_loc=self.cache_loc,
            spec_info_padded=SimpleNamespace(allocate_lens=np.array([12, 8, 0])),
        )
        mds = self.backend.get_eagle_multi_step_metadata(batch)
        self.assertEqual(len(mds), 3)
        for i, md in enumerate(mds):
            self._check_virtual(md, [7 + i, 4 + i, 0], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
