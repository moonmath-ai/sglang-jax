"""Vectorized topk=1 chain accept: target-only sampling == tree_speculative_sampling_target_only,
greedy == the sequential greedy chain rule.

    JAX_PLATFORMS=cpu python -m pytest python/sgl_jax/test/speculative/test_chain_fused_sampling.py
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.speculative.kernel import tree_speculative_sampling_target_only
from sgl_jax.srt.speculative.chain_fused_round import _accept_greedy, _accept_sampling
from sgl_jax.srt.speculative.eagle_util import build_chain_verify_inputs


class TestChainFusedSampling(unittest.TestCase):
    def test_matches_reference(self):
        bs, T, V = 6, 4, 11
        rng = np.random.default_rng(0)
        for trial in range(20):
            logits = jnp.asarray(rng.standard_normal((bs * T, V)) * 2.0, jnp.float32)
            temps = jnp.full((bs,), 0.7, jnp.float32)
            top_ks = jnp.full((bs,), V, jnp.int32)
            top_ps = jnp.ones((bs,), jnp.float32)
            # Make draft tokens often equal to the argmax so acceptance varies.
            am = np.asarray(jnp.argmax(logits, -1)).reshape(bs, T)
            drafts = np.where(
                rng.random((bs, T - 1)) < 0.7, am[:, :-1], rng.integers(0, V, (bs, T - 1))
            )
            verified = rng.integers(0, V, bs)
            packed = build_chain_verify_inputs(verified, drafts, np.zeros(bs, np.int32), T, bs)
            draft_tokens = jnp.asarray(packed[0])
            key = jax.random.PRNGKey(trial)

            acc_idx, n_acc, predict = _accept_sampling(
                logits,
                draft_tokens,
                temps,
                top_ks,
                top_ps,
                key,
                T=T,
                use_top_p=False,
                threshold_single=1.0,
                threshold_acc=1.0,
            )

            # Reference with the same coins.
            k1, k2 = jax.random.split(key)
            coins = jax.random.uniform(k1, (bs, T), dtype=jnp.float32)
            coin_f = jax.random.uniform(k2, (bs,), dtype=jnp.float32)
            probs = jax.nn.softmax(logits / jnp.repeat(temps, T)[:, None], axis=-1)
            r_idx, r_n, r_pred = tree_speculative_sampling_target_only(
                predicts=jnp.zeros((bs * T + 1,), jnp.int32),
                accept_index=jnp.full((bs, T), -1, jnp.int32),
                accept_token_num=jnp.zeros((bs,), jnp.int32),
                candidates=draft_tokens.reshape(bs, T),
                retrive_index=jnp.asarray(packed[2]).reshape(bs, T),
                retrive_next_token=jnp.asarray(packed[3]).reshape(bs, T),
                retrive_next_sibling=jnp.asarray(packed[4]).reshape(bs, T),
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coin_f,
                target_probs=probs,
                draft_probs=jnp.zeros_like(probs),
            )
            np.testing.assert_array_equal(np.asarray(n_acc), np.asarray(r_n))
            np.testing.assert_array_equal(np.asarray(acc_idx), np.asarray(r_idx))
            np.testing.assert_array_equal(
                np.asarray(predict)[: bs * T], np.asarray(r_pred)[: bs * T]
            )

    def test_greedy_matches_sequential_rule(self):
        bs, T, V = 6, 4, 11
        rng = np.random.default_rng(1)
        for _ in range(20):
            logits = rng.standard_normal((bs * T, V)).astype(np.float32)
            target = logits.argmax(-1).reshape(bs, T)
            drafts = np.where(
                rng.random((bs, T - 1)) < 0.7, target[:, :-1], rng.integers(0, V, (bs, T - 1))
            )
            cand = np.concatenate([rng.integers(0, V, (bs, 1)), drafts], axis=1)
            acc_idx, n_acc, predict = _accept_greedy(
                jnp.asarray(logits), jnp.asarray(cand.reshape(-1), jnp.int32), T=T
            )
            for r in range(bs):
                n = 0
                while n < T - 1 and cand[r, n + 1] == target[r, n]:
                    n += 1
                self.assertEqual(int(n_acc[r]), n)
                expect_idx = [r * T + k if k <= n else -1 for k in range(T)]
                np.testing.assert_array_equal(np.asarray(acc_idx[r]), expect_idx)
                # Accepted drafts, then the target's token at the stop row.
                np.testing.assert_array_equal(
                    np.asarray(predict)[r * T : r * T + n + 1], target[r, : n + 1]
                )


if __name__ == "__main__":
    unittest.main()
