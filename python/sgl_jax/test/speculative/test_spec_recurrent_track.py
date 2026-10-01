"""Extra-buffer radix cache under speculative decoding: the commit snapshots the state at the
crossed track boundary (device) and the scheduler records it and flips the slot (host)."""

from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.schedule_batch import record_spec_recurrent_track
from sgl_jax.srt.mem_cache.recurrent_state_pool import _commit_speculative_state

SLOTS, B, T, H, D = 9, 3, 4, 2, 8


def test_commit_snapshots_track_step():
    mesh = jax.make_mesh((1, 1), ("data", "tensor"), devices=jax.devices()[:1])
    rng = np.random.default_rng(0)
    ssm = np.asarray(rng.normal(size=(SLOTS, H, D, D)), np.float32)
    conv = np.asarray(rng.normal(size=(SLOTS, H * D, 3)), np.float32)
    spec_ssm = np.asarray(rng.normal(size=(B, T, H, D, D)), np.float32)
    spec_conv = np.asarray(rng.normal(size=(B, T, H * D, 3)), np.float32)
    idx = np.asarray([1, 2, 3], np.int32)
    acc = np.asarray([4, 2, 3], np.int32)
    t_idx = np.asarray([5, 6, 7], np.int32)
    t_step = np.asarray([1, -1, 2], np.int32)  # row 1 crossed nothing
    put = lambda x, *spec: jax.device_put(x, NamedSharding(mesh, P(*spec)))  # noqa: E731
    with jax.set_mesh(mesh):
        (new_ssm,), ((new_conv, _, _),) = _commit_speculative_state(
            [put(ssm, "data", "tensor")],
            [
                [
                    put(conv, "data", "tensor"),
                    put(spec_conv, "data", None, "tensor"),
                    put(spec_ssm, "data", None, "tensor"),
                ]
            ],
            *(put(x, "data") for x in (idx, acc)),
            mesh=mesh,
            data_axis="data",
            recurrent_axis="tensor",
            conv_axis="tensor",
            track_indices=put(t_idx, "data"),
            track_steps=put(t_step, "data"),
        )
    for got, pool, spec in ((new_ssm, ssm, spec_ssm), (new_conv, conv, spec_conv)):
        got = np.asarray(got)
        for b in range(B):
            np.testing.assert_array_equal(got[idx[b]], spec[b, acc[b] - 1])
        np.testing.assert_array_equal(got[5], spec[0, 1])
        np.testing.assert_array_equal(got[7], spec[2, 2])
        np.testing.assert_array_equal(got[[4, 6, 8]], pool[[4, 6, 8]])


def _req(prompt_len, n_out):
    return SimpleNamespace(
        origin_input_ids=[0] * prompt_len,
        output_ids=[0] * n_out,
        recurrent_next_track_idx=0,
        recurrent_last_track_seqlen=None,
    )


POOL = SimpleNamespace(get_recurrent_ping_pong_other_idx=lambda i: (i + 1) % 2)


def test_host_records_crossed_boundary():
    req = _req(100, 26)  # state 125 tokens before; +4 -> 129 crosses 128
    old = len(req.origin_input_ids) + len(req.output_ids) - 1
    req.output_ids += [0] * 4
    record_spec_recurrent_track(req, old, 4, interval=128, pool=POOL)
    assert req.recurrent_last_track_seqlen == 128 and req.recurrent_next_track_idx == 1


def test_host_skips_uncrossed_or_truncated():
    req = _req(100, 20)
    old = len(req.origin_input_ids) + len(req.output_ids) - 1
    req.output_ids += [0] * 4
    record_spec_recurrent_track(req, old, 4, interval=128, pool=POOL)
    assert req.recurrent_last_track_seqlen is None and req.recurrent_next_track_idx == 0

    req = _req(100, 26)  # crosses 128, but a stop token truncated the output to 127 tokens
    old = len(req.origin_input_ids) + len(req.output_ids) - 1
    req.output_ids += [0]
    record_spec_recurrent_track(req, old, 4, interval=128, pool=POOL)
    assert req.recurrent_last_track_seqlen is None and req.recurrent_next_track_idx == 0
