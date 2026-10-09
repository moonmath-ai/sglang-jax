"""Speculative commit of the KDA verify scratch: each request's running state becomes its
verify state after the accepted tokens; rows that must not move (padding slot 0, accept <= 0)
leave their running slot untouched; a crossed track boundary also snapshots into the track
slot (extra-buffer radix cache).

    JAX_PLATFORMS=cpu python -m pytest python/sgl_jax/test/speculative/test_recurrent_spec_commit.py
"""

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.recurrent_state_pool import _commit_speculative_state

SLOTS, T, H, D, KM1 = 10, 4, 2, 8, 3


def test_commit_accepted_step_padding_and_track():
    mesh = jax.make_mesh((1, 1), ("data", "tensor"), devices=jax.devices()[:1])
    rng = np.random.default_rng(0)
    ssm = rng.normal(size=(SLOTS, H, D, D)).astype(np.float32)
    conv = rng.normal(size=(SLOTS, KM1, 3, H, D)).astype(np.float32)  # tiled conv layout
    # Batch rows: 3 live requests, one with accept 0, one padding row (slot 0).
    idx = np.asarray([1, 2, 3, 4, 0], np.int32)
    acc = np.asarray([4, 2, 3, 0, 2], np.int32)
    t_idx = np.asarray([6, 7, 8, 9, 0], np.int32)
    t_step = np.asarray([1, -1, 2, -1, -1], np.int32)  # rows 0 and 2 crossed a boundary
    B = len(idx)
    spec_ssm = rng.normal(size=(B, T, H, D, D)).astype(np.float32)
    spec_conv = rng.normal(size=(B, T, KM1, 3, H, D)).astype(np.float32)

    ssm_spec, spec_ssm_spec = P("data", "tensor", None, None), P("data", None, "tensor", None, None)
    conv_spec = P("data", None, None, "tensor", None)
    spec_conv_spec = P("data", None, None, None, "tensor", None)
    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec))  # noqa: E731
    with jax.set_mesh(mesh):
        (new_ssm,), ((new_conv, _, _),) = _commit_speculative_state(
            [put(ssm, ssm_spec)],
            [[put(conv, conv_spec), put(spec_conv, spec_conv_spec), put(spec_ssm, spec_ssm_spec)]],
            *(put(x, P("data")) for x in (idx, acc)),
            mesh=mesh,
            data_axis="data",
            buffer_specs=(ssm_spec, spec_ssm_spec, conv_spec, spec_conv_spec),
            track_indices=put(t_idx, P("data")),
            track_steps=put(t_step, P("data")),
        )
    for got, pool, spec in ((new_ssm, ssm, spec_ssm), (new_conv, conv, spec_conv)):
        got = np.asarray(got)
        for b in range(3):  # accepted rows: state after acc tokens (step acc - 1)
            np.testing.assert_array_equal(got[idx[b]], spec[b, acc[b] - 1])
        np.testing.assert_array_equal(got[4], pool[4])  # accept 0: running state kept
        np.testing.assert_array_equal(got[6], spec[0, 1])  # track snapshots
        np.testing.assert_array_equal(got[8], spec[2, 2])
        np.testing.assert_array_equal(got[[5, 7, 9]], pool[[5, 7, 9]])  # untouched slots
