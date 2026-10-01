"""KDA speculative target verify: TARGET_VERIFY over a T-token chain + commit must match
T sequential DECODE steps (outputs), and the committed state must equal the decode state
after exactly the accepted number of tokens.

    JAX_PLATFORMS=cpu python -m pytest python/sgl_jax/test/speculative/test_kda_spec_verify.py
"""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

H, D, K, T = 2, 16, 4, 4
HIDDEN = H * D


def _pool(mesh, spec_T, tiled=False, lazy=False):
    return RecurrentStatePool(
        linear_recurrent_layer_ids=[0],
        size=4,
        num_heads=H,
        head_dim=D,
        conv_kernel_size=K,
        mesh=mesh,
        temporal_dtype=jnp.float32,
        conv_dtype=jnp.float32,
        spec_num_draft_tokens=spec_T,
        conv_tiled=tiled,
        lazy_commit=lazy,
    )


def _to_tiled(conv):  # [slots, 3*H*D, K-1] -> [slots, K-1, 3, H, D]
    return np.swapaxes(conv, 1, 2).reshape(conv.shape[0], K - 1, 3, H, D)


def _to_legacy(conv):
    return np.swapaxes(np.asarray(conv).reshape(conv.shape[0], K - 1, -1), 1, 2)


class TestKDASpecVerify(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
        rng = np.random.default_rng(0)
        f = lambda *s, scale=0.3: jnp.asarray(rng.standard_normal(s) * scale, jnp.float32)
        self.layer = RadixLinearAttention(
            layer_id=0,
            num_q_heads=H,
            num_k_heads=H,
            num_v_heads=H,
            head_q_dim=D,
            head_k_dim=D,
            head_v_dim=D,
            q_conv1d=SimpleNamespace(weight=SimpleNamespace(value=f(HIDDEN, K, scale=1.0))),
            k_conv1d=SimpleNamespace(weight=SimpleNamespace(value=f(HIDDEN, K, scale=1.0))),
            v_conv1d=SimpleNamespace(weight=SimpleNamespace(value=f(HIDDEN, K, scale=1.0))),
            activation="silu",
            A_log=SimpleNamespace(value=f(1, 1, H, 1, scale=1.0)),
            dt_bias=SimpleNamespace(value=f(H * D, scale=1.0)),
            scale=D**-0.5,
            kda_lower_bound=-5.0,
        )
        # 2 live requests (slots 1, 2) + one padding row (slot 0).
        self.indices = np.array([1, 2, 0], np.int32)
        self.bs = len(self.indices)
        self.ssm0 = f(6, H, D, D)
        self.conv0 = f(6, 3 * HIDDEN, K - 1)
        n = self.bs * T
        self.q, self.k, self.v, self.a = (f(n, HIDDEN) for _ in range(4))
        self.b = jax.nn.sigmoid(f(n, H, scale=1.0))

    def _init(self, pool):
        extra = pool.conv_buffers[0][1:]
        conv0 = jnp.asarray(_to_tiled(np.asarray(self.conv0))) if pool.conv_tiled else self.conv0
        pool.replace_buffer(([self.ssm0], [[conv0] + list(extra)]))

    def _meta(self, cu):
        sh = NamedSharding(self.mesh, P("data"))
        return LinearRecurrentAttnBackendMetadata(
            cu_q_lens=jax.device_put(cu, sh),
            recurrent_indices=jax.device_put(self.indices, sh),
            has_initial_state=jax.device_put(np.ones(self.bs, bool), sh),
        )

    def _run(self, backend, pool, mode, q, k, v, a, b):
        out, (ssm, conv_list) = backend(
            q, k, v, a, b, self.layer, SimpleNamespace(forward_mode=mode), pool
        )
        pool.replace_buffer(([ssm], [conv_list]))
        return out

    def test_verify_matches_sequential_decode(self):
        self._check(tiled=False)

    def test_verify_matches_sequential_decode_tiled_conv(self):
        self._check(tiled=True)

    def _check(self, tiled):
        with jax.set_mesh(self.mesh):
            backend = KDAAttnBackend(mesh=self.mesh)

            # Reference: T sequential decode steps.
            ref_pool = _pool(self.mesh, 0, tiled=tiled)  # DECODE through the tiled layout too
            self._init(ref_pool)
            ref_conv_of = (lambda c: _to_legacy(c)) if tiled else (lambda c: np.asarray(c))
            backend.forward_metadata = self._meta(np.arange(self.bs + 1, dtype=np.int32))
            tok = lambda x, t: x.reshape(self.bs, T, *x.shape[1:])[:, t]
            ref_out, ref_ssm, ref_conv = [], [], []
            for t in range(T):
                ref_out.append(
                    self._run(
                        backend,
                        ref_pool,
                        ForwardMode.DECODE,
                        *(tok(x, t) for x in (self.q, self.k, self.v, self.a, self.b)),
                    )
                )
                ref_ssm.append(np.asarray(ref_pool.recurrent_buffers[0]))
                ref_conv.append(ref_conv_of(ref_pool.conv_buffers[0][0]))
            ref_out = np.stack([np.asarray(o) for o in ref_out], axis=1).reshape(self.bs * T, -1)

            # Verify: one TARGET_VERIFY forward, running state untouched.
            pool = _pool(self.mesh, T, tiled=tiled)
            self._init(pool)
            conv_of = (lambda c: _to_legacy(c)) if tiled else (lambda c: np.asarray(c))
            backend.forward_metadata = self._meta(np.arange(self.bs + 1, dtype=np.int32) * T)
            out = self._run(
                backend, pool, ForwardMode.TARGET_VERIFY, self.q, self.k, self.v, self.a, self.b
            )
            live = slice(0, 2 * T)  # padding row output is garbage by design
            # fused kernel: bf16-rounded conv / q, k like XLA; on TPU XLA's default matmul
            # precision also differs between the decode and verify reference paths.
            tol = 2e-2 if tiled or jax.default_backend() == "tpu" else 1e-5
            np.testing.assert_allclose(np.asarray(out)[live], ref_out[live], rtol=tol, atol=tol)
            ssm0, conv0 = np.asarray(self.ssm0), np.asarray(self.conv0)
            np.testing.assert_array_equal(np.asarray(pool.recurrent_buffers[0]), ssm0)
            np.testing.assert_array_equal(conv_of(pool.conv_buffers[0][0]), conv0)

            # Commit: req0 accepted 3 tokens, req1 accepted 1; padding row (slot 0) ignored.
            accept = np.array([3, 1, 4], np.int32)
            sh = NamedSharding(self.mesh, P("data"))
            pool.commit_speculative_state(
                jax.device_put(self.indices, sh), jax.device_put(accept, sh)
            )
            ssm = np.asarray(pool.recurrent_buffers[0])
            conv = conv_of(pool.conv_buffers[0][0])
            stol = 2e-2 if tiled or jax.default_backend() == "tpu" else 1e-6
            for row, slot in enumerate(self.indices[:2]):
                step = accept[row] - 1
                np.testing.assert_allclose(ssm[slot], ref_ssm[step][slot], rtol=stol, atol=stol)
                np.testing.assert_allclose(conv[slot], ref_conv[step][slot], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(ssm[3:], ssm0[3:])  # slot 0 is the dummy sink

    def test_lazy_commit_two_rounds(self):
        """Slot-indexed scratch + recorded accepted step: round 2 must seed from round 1's
        accepted scratch step, and the effective state must track sequential decode."""
        rng = np.random.default_rng(7)
        f = lambda *s_, scale=0.3: jnp.asarray(rng.standard_normal(s_) * scale, jnp.float32)
        n = self.bs * T
        r2 = [f(n, HIDDEN) for _ in range(4)] + [jax.nn.sigmoid(f(n, H, scale=1.0))]
        r1 = [self.q, self.k, self.v, self.a, self.b]
        acc1, acc2 = 2, 3
        sh = NamedSharding(self.mesh, P("data"))
        tok = lambda x, t: x.reshape(self.bs, T, *x.shape[1:])[:, t]
        ssm0_np = np.asarray(self.ssm0)
        with jax.set_mesh(self.mesh):
            backend = KDAAttnBackend(mesh=self.mesh)
            # reference: decode acc1 tokens of round 1, then all T tokens of round 2
            ref_pool = _pool(self.mesh, 0)
            self._init(ref_pool)
            backend.forward_metadata = self._meta(np.arange(self.bs + 1, dtype=np.int32))
            for t in range(acc1):
                self._run(backend, ref_pool, ForwardMode.DECODE, *(tok(x, t) for x in r1))
            ref_out, ref_ssm = [], []
            for t in range(T):
                ref_out.append(
                    np.asarray(
                        self._run(backend, ref_pool, ForwardMode.DECODE, *(tok(x, t) for x in r2))
                    )
                )
                ref_ssm.append(np.asarray(ref_pool.recurrent_buffers[0]))
            ref_out = np.stack(ref_out, axis=1).reshape(n, -1)

            pool = _pool(self.mesh, T, tiled=True, lazy=True)
            self.ssm0 = jnp.asarray(
                ssm0_np
            )  # the ref pool's commit-free run kept it alive; re-wrap
            self._init(pool)
            backend.forward_metadata = self._meta(np.arange(self.bs + 1, dtype=np.int32) * T)
            idx = jax.device_put(self.indices, sh)
            self._run(backend, pool, ForwardMode.TARGET_VERIFY, *r1)
            pool.commit_speculative_state(
                idx, jax.device_put(np.array([acc1, acc1, 4], np.int32), sh)
            )
            out2 = self._run(backend, pool, ForwardMode.TARGET_VERIFY, *r2)
            live = slice(0, 2 * T)
            np.testing.assert_allclose(np.asarray(out2)[live], ref_out[live], rtol=2e-2, atol=2e-2)
            pool.commit_speculative_state(
                idx, jax.device_put(np.array([acc2, acc2, 4], np.int32), sh)
            )
            steps = np.asarray(pool.conv_buffers[0][3])
            scratch = np.asarray(pool.conv_buffers[0][2])
            for slot in self.indices[:2]:
                self.assertEqual(steps[slot], acc2 - 1)
                np.testing.assert_allclose(
                    scratch[slot, steps[slot]], ref_ssm[acc2 - 1][slot], rtol=2e-2, atol=2e-2
                )
            # the pool's running rows are untouched under lazy commit
            np.testing.assert_array_equal(np.asarray(pool.recurrent_buffers[0]), ssm0_np)


if __name__ == "__main__":
    unittest.main()
