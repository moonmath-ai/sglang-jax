"""Shift-and-accumulate EXTEND conv == windowed-gather reference (short_convolution).

    JAX_PLATFORMS=cpu python -m pytest python/sgl_jax/test/test_short_conv_extend_shift.py
"""

import unittest

import jax.numpy as jnp
import numpy as np

import sgl_jax.srt.layers.attention.linear.short_convolution as sc
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def _run(shift, x, w, cache, cu, bias, act):
    old = sc._EXTEND_CONV_SHIFT
    sc._EXTEND_CONV_SHIFT = shift
    try:
        return sc.short_convolution(x, w, cache, cu, ForwardMode.EXTEND, bias=bias, activation=act)
    finally:
        sc._EXTEND_CONV_SHIFT = old


class TestExtendConvShift(unittest.TestCase):
    def test_matches_windowed_gather(self):
        rng = np.random.default_rng(0)
        D, K = 32, 4
        cases = [
            [5, 1, 2, 7],  # sequences shorter than K-1
            [0, 3, 0, 9, 1],  # zero-length (padding) sequences
            [16],
            [2, 2, 2, 2, 2, 2],
        ]
        for lens in cases:
            for pad in (0, 5):  # trailing padded tokens beyond cu[-1]
                for bias, act in ((None, "silu"), (True, None)):
                    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
                    T = int(cu[-1]) + pad
                    x = jnp.asarray(rng.standard_normal((T, D)), jnp.float32)
                    w = jnp.asarray(rng.standard_normal((D, K)), jnp.float32)
                    cache = jnp.asarray(rng.standard_normal((len(lens), D, K - 1)), jnp.float32)
                    b = jnp.asarray(rng.standard_normal((D,)), jnp.float32) if bias else None
                    y0, c0 = _run(False, x, w, cache, jnp.asarray(cu), b, act)
                    y1, c1 = _run(True, x, w, cache, jnp.asarray(cu), b, act)
                    n = int(cu[-1])
                    np.testing.assert_allclose(
                        np.asarray(y1)[:n],
                        np.asarray(y0)[:n],
                        rtol=1e-5,
                        atol=1e-5,
                        err_msg=str(lens),
                    )
                    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c0))


if __name__ == "__main__":
    unittest.main()
