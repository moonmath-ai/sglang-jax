"""One-JIT speculative decode round for topk=1 chains on non-FlashAttention backends.

The generic EAGLE loop (``EagleDraftWorker.draft`` -> ``BaseSpecWorker.verify`` ->
``EagleDraftWorker.draft_extend_for_decode``) issues four model dispatches plus ~60 small
eager ops, ~20 metadata uploads and several host syncs per round; on a TP-8 TPU mesh that
host overhead is several times the device time. This module runs a whole round in one JIT:

    draft steps (num_steps - 1 forwards) -> chain -> target TARGET_VERIFY -> accept
    (greedy / target-only sampling) -> recurrent-state commit (hybrid targets)
    -> accepted-row gather -> draft DRAFT_EXTEND -> next round's top-1

Every phase's attention metadata depends only on ``seq_lens`` / ``allocate_lens`` (the
backends express each spec forward as a decode batch, see
``MLAAttentionBackend.get_eagle_forward_metadata``), so it is built on the host up front and
uploaded with one ``device_put``. Results come back with one ``device_get``. Outputs match the
generic loop's ``GenerationBatchResult`` so the scheduler is unchanged.

Scope: NEXTN / EAGLE with one draft model, topk == 1, num_draft_tokens == num_steps + 1,
dp_size == 1, MLA-family attention on both models, no logprobs.
"""
from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.speculative.kernel import top_k_renorm_prob, top_p_renorm_prob
from sgl_jax.srt.kernels.speculative.verify_tree_greedy_kernel import verify_tree_greedy
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend, MLAAttentionMetadata
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import global_server_args_dict
from sgl_jax.srt.mem_cache.recurrent_state_pool import _commit_speculative_state
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.eagle_util import build_chain_verify_inputs_device
from sgl_jax.srt.speculative.spec_utils import (
    SIMULATED_ACCEPTANCE_CONFIG,
    apply_simulated_acceptance,
)
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


def _mla_backend(backend):
    if isinstance(backend, HybridLinearAttnBackend):
        backend = backend.full_attn_backend
    return backend if isinstance(backend, MLAAttentionBackend) else None


def can_use_chain_fused_round(spec_worker) -> bool:
    sa = spec_worker.server_args
    return (
        spec_worker.topk == 1
        and spec_worker.speculative_num_steps > 1
        and spec_worker.speculative_num_draft_tokens == spec_worker.speculative_num_steps + 1
        and sa.dp_size == 1
        and not hasattr(spec_worker.draft_worker, "_workers")
        and _mla_backend(spec_worker.target_worker.model_runner.attn_backend) is not None
        and _mla_backend(spec_worker.draft_worker.draft_model_runner.attn_backend) is not None
    )


def _run_model(model_def, state_def, leaves, forward_batch, pools, logits_metadata):
    model = nnx.merge(model_def, jax.tree_util.tree_unflatten(state_def, leaves))
    output, pool_updates, _, _ = model(forward_batch, pools, logits_metadata)
    pools.replace_all(pool_updates)
    return output, pool_updates


def _accept_greedy(logits, draft_tokens, retrive_index, retrive_next_token,
                   retrive_next_sibling, *, steps, T):
    return verify_tree_greedy(
        speculative_num_steps=steps,
        num_draft_tokens=T,
        draft_tokens=draft_tokens,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        next_token_logits=logits,
    )


def _accept_sampling(logits, draft_tokens, temperatures, top_ks, top_ps, key, *, steps, T,
                     use_top_p, threshold_single, threshold_acc):
    """Target-only speculative sampling for a topk=1 chain, vectorized.

    Same rule as ``tree_speculative_sampling_target_only`` (draft probs treated as 0):
    accept d_j while ``u[j-1] <= p_{j-1}(d_j) / threshold_acc`` or
    ``p_{j-1}(d_j) >= threshold_single``; the bonus token is then sampled from the target
    distribution at the stop row, with the rejected draft token's mass removed.
    Returns (accept_index [bs, T], accept_length (excl. bonus) [bs], predict [bs*T+1]).
    """
    bs = draft_tokens.shape[0] // T
    temp = jnp.repeat(temperatures, T)[:, None]
    probs = jax.nn.softmax(logits.astype(jnp.float32) / temp, axis=-1)
    probs = top_k_renorm_prob(probs, jnp.repeat(top_ks, T))
    probs = jax.lax.cond(
        use_top_p,
        lambda x: top_p_renorm_prob(x, jnp.repeat(top_ps, T)),
        lambda x: x,
        probs,
    )
    probs = probs.reshape(bs, T, -1)
    cand = draft_tokens.reshape(bs, T)
    k1, k2 = jax.random.split(key)
    coins = jax.random.uniform(k1, (bs, T), dtype=jnp.float32)
    coin_final = jax.random.uniform(k2, (bs,), dtype=jnp.float32)

    # p_{j-1}(d_j), j = 1..T-1
    p_draft = jnp.take_along_axis(probs[:, : T - 1], cand[:, 1:, None], axis=-1)[..., 0]
    ok = (coins[:, : T - 1] <= p_draft / threshold_acc) | (p_draft >= threshold_single)
    n_acc = jnp.sum(jnp.cumprod(ok.astype(jnp.int32), axis=1), axis=1).astype(jnp.int32)

    rows = jnp.arange(bs)
    q = probs[rows, n_acc]  # [bs, V]
    rejected = cand[rows, jnp.minimum(n_acc + 1, T - 1)]
    q = jnp.where(
        (n_acc < T - 1)[:, None] & (jnp.arange(q.shape[-1])[None, :] == rejected[:, None]), 0.0, q
    )
    cdf = jnp.cumsum(q, axis=-1)
    bonus = jnp.argmax(cdf > (coin_final * cdf[:, -1])[:, None], axis=-1).astype(jnp.int32)

    j = jnp.arange(T, dtype=jnp.int32)[None, :]
    base = (rows * T)[:, None].astype(jnp.int32)
    accept_index = jnp.where(j <= n_acc[:, None], base + j, -1).astype(jnp.int32)
    # predict[r*T + k] = d_{k+1} for k < n_acc, predict[r*T + n_acc] = bonus
    nxt = jnp.concatenate([cand[:, 1:], jnp.zeros((bs, 1), cand.dtype)], axis=1)
    pred = jnp.where(j < n_acc[:, None], nxt, jnp.where(j == n_acc[:, None], bonus[:, None], 0))
    predict = jnp.concatenate([pred.reshape(-1), jnp.zeros((1,), jnp.int32)]).astype(jnp.int32)
    return accept_index, n_acc, predict


def _unpack(blob, layout, name):
    off, shape = layout[name]
    n = int(np.prod(shape)) if shape else 1
    x = jax.lax.slice_in_dim(blob, off, off + n)
    return x.reshape(shape) if shape else x[0]


def _fused_round(
    tgt_def, tgt_state_def, tgt_leaves, tgt_pools,
    dr_def, dr_state_def, dr_leaves, dr_pools,
    elm, int_blob, f32_blob, hidden, base_key,
    *, layout, steps, T, bs, threshold_single, threshold_acc, mesh, tgt_backend_def,
    dr_backend_def,
):
    rep = NamedSharding(mesh, P())
    data = NamedSharding(mesh, P("data"))
    I = lambda name: _unpack(int_blob, layout, name)  # noqa: E731
    F = lambda name: _unpack(f32_blob, layout, name)  # noqa: E731
    D = lambda name: jax.sharding.reshard(I(name), data)  # noqa: E731

    def mla_md(prefix):
        return MLAAttentionMetadata(*(
            D(prefix + k) if (prefix + k) in layout else None for k in _MD_KEYS))

    seq_lens = I("seq_lens")
    verified_id = I("verified_id")
    first_draft_token = I("first_draft")
    recurrent_indices = I("recurrent_indices")
    extend_seq_lens = D("extend_seq_lens")
    extend_logits_indices = D("extend_logits_indices")
    temperatures, top_ps = F("temperatures"), F("top_ps")
    top_ks = I("top_ks")
    is_greedy, use_top_p = I("is_greedy")[0] != 0, I("use_top_p")[0] != 0
    key = jax.random.fold_in(base_key, I("round_id")[0])
    dr_step_mds = [mla_md(f"step{i}_") for i in range(steps - 1)]
    dr_extend_md = mla_md("extend_")

    tgt_backend = jax.tree_util.tree_unflatten(tgt_backend_def, [])
    if isinstance(tgt_backend, HybridLinearAttnBackend):
        tgt_backend.forward_metadata = HybridLinearAttnBackendMetadata(
            full_attn_metadata=mla_md("verify_"),
            linear_attn_metadata=LinearRecurrentAttnBackendMetadata(
                cu_q_lens=D("kda_cu_q"), recurrent_indices=D("recurrent_indices"),
                has_initial_state=D("has_initial_state") != 0),
        )
    else:
        tgt_backend.forward_metadata = mla_md("verify_")
    dr_backend = jax.tree_util.tree_unflatten(dr_backend_def, [])

    def fb(mode, n, backend):
        return ForwardBatch(bid=0, forward_mode=mode, batch_size=n, input_ids=None,
                            req_pool_indices=None, seq_lens=None, out_cache_loc=None,
                            attn_backend=backend, expert_location_metadata=elm,
                            capture_hidden_mode=CaptureHiddenMode.FULL)

    tgt_fb_verify = fb(ForwardMode.TARGET_VERIFY, bs * T, tgt_backend)
    dr_fb_step = fb(ForwardMode.DECODE, bs, dr_backend)
    dr_fb_extend = fb(ForwardMode.DRAFT_EXTEND, bs * T, dr_backend)

    # ---- draft: num_steps - 1 decode forwards of the MTP block ----
    tokens = [first_draft_token]
    tok, h = first_draft_token, hidden
    lm_step = LogitsMetadata(forward_mode=ForwardMode.DECODE,
                             capture_hidden_mode=CaptureHiddenMode.LAST)
    for i in range(steps - 1):
        dr_fb_step.attn_backend.forward_metadata = dr_step_mds[i]
        dr_fb_step.input_ids = tok
        dr_fb_step.positions = seq_lens + i
        dr_fb_step.spec_info = EagleDraftInput(hidden_states=h)
        out, _ = _run_model(dr_def, dr_state_def, dr_leaves, dr_fb_step, dr_pools, lm_step)
        logits = jax.sharding.reshard(out.next_token_logits, rep)
        tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        h = jax.sharding.reshard(out.hidden_states, rep)
        tokens.append(tok)
    token_list = jnp.stack(tokens, axis=1)  # [bs, steps]

    # ---- target verify over [bonus, d1, .., d_{T-1}] ----
    valid = seq_lens > 0
    verified_seq_lens = jnp.where(valid, seq_lens - 1, 0)
    packed = build_chain_verify_inputs_device(verified_id, token_list, verified_seq_lens, T, bs)
    packed = jax.sharding.reshard(packed, rep)
    draft_tokens, positions = packed[0], packed[1]
    retrive_index = packed[2].reshape(bs, T)
    retrive_next_token = packed[3].reshape(bs, T)
    retrive_next_sibling = packed[4].reshape(bs, T)
    tgt_fb_verify.input_ids = draft_tokens
    tgt_fb_verify.positions = positions
    lm_verify = LogitsMetadata(forward_mode=ForwardMode.TARGET_VERIFY,
                               capture_hidden_mode=CaptureHiddenMode.FULL)
    out, tgt_updates = _run_model(tgt_def, tgt_state_def, tgt_leaves, tgt_fb_verify, tgt_pools,
                                  lm_verify)
    logits = jax.sharding.reshard(out.next_token_logits, rep)
    target_hidden = jax.sharding.reshard(out.hidden_states, rep)

    # One executable per bucket: greedy vs sampling (and top-p) are runtime flags.
    accept_index, accept_length, predict = jax.lax.cond(
        is_greedy,
        lambda: _accept_greedy(
            logits, draft_tokens, retrive_index, retrive_next_token, retrive_next_sibling,
            steps=steps, T=T),
        lambda: _accept_sampling(
            logits, draft_tokens, temperatures, top_ks, top_ps, key, steps=steps, T=T,
            use_top_p=use_top_p, threshold_single=threshold_single,
            threshold_acc=threshold_acc),
    )
    accept_length = (accept_length + 1).astype(jnp.int32)
    if SIMULATED_ACCEPTANCE_CONFIG.enabled:
        # SIMULATE_ACC_LEN: throughput-mechanism benchmarks with a fixed accept length
        # (bonus included), same semantics as EagleVerifyInput.sample.
        cfg = SIMULATED_ACCEPTANCE_CONFIG
        target_predict = (
            jnp.argmax(logits, axis=-1).astype(jnp.int32).reshape(bs, T)
            if cfg.token_mode == "real-draft-token"
            else None
        )
        accept_index, predict, accept_length = apply_simulated_acceptance(
            accept_index=accept_index,
            predict=predict,
            accept_lens=accept_length,
            candidates=draft_tokens.reshape(bs, T),
            target_predict=target_predict,
            valid_mask=valid,
            spec_steps=steps,
            topk=1,
            rng=jax.random.fold_in(key, 1),
        )
        accept_length = accept_length.astype(jnp.int32)

    # ---- hybrid target: keep the recurrent state after the accepted prefix ----
    if "recurrent_state_pool" in tgt_pools._pools:
        rsp = tgt_pools.recurrent_state_pool
        data = NamedSharding(mesh, P(rsp.data_partition_axis))
        new_rec, new_conv = _commit_speculative_state(
            rsp.recurrent_buffers, rsp.conv_buffers,
            jax.sharding.reshard(recurrent_indices, data),
            jax.sharding.reshard(accept_length, data),
            mesh=mesh, data_axis=rsp.data_partition_axis,
            recurrent_axis=rsp.recurrent_partition_axis, conv_axis=rsp.conv_partition_axis)
        tgt_updates = dict(tgt_updates)
        tgt_updates["recurrent_state_pool"] = (new_rec, new_conv)

    # ---- gather accepted rows (rejected -> the request's last slot) ----
    accept_index = accept_index.reshape(-1)  # [bs * T]
    req_last = (jnp.arange(bs * T, dtype=jnp.int32) // T) * T + (T - 1)
    safe_index = jnp.where(accept_index >= 0, accept_index, req_last)
    verified_flat = jnp.where(accept_index >= 0, predict[jnp.maximum(accept_index, 0)], 0)
    verified_flat = verified_flat.astype(jnp.int32)
    ext_hidden = target_hidden[safe_index]
    ext_positions = positions[safe_index]
    new_seq_lens = verified_seq_lens + accept_length + 1

    # ---- draft extend over the verified chain ----
    dr_fb_extend.attn_backend.forward_metadata = dr_extend_md
    dr_fb_extend.input_ids = verified_flat
    dr_fb_extend.positions = ext_positions
    dr_fb_extend.spec_info = EagleDraftInput(hidden_states=ext_hidden)
    lm_extend = LogitsMetadata(
        forward_mode=ForwardMode.DRAFT_EXTEND,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        extend_seq_lens=extend_seq_lens,
        logits_indices=extend_logits_indices,
        accept_lens=accept_length,
    )
    out, dr_updates = _run_model(dr_def, dr_state_def, dr_leaves, dr_fb_extend, dr_pools,
                                 lm_extend)
    ext_logits = jax.sharding.reshard(out.next_token_logits, rep)
    ext_hidden_all = jax.sharding.reshard(out.hidden_states, rep)
    select_index = jnp.arange(bs, dtype=jnp.int32) * T + jnp.maximum(accept_length - 1, 0)
    next_hidden = ext_hidden_all[select_index]
    next_verified_id = verified_flat[select_index]
    lse = jax.nn.logsumexp(ext_logits.astype(jnp.float32), axis=-1, keepdims=True)
    top_logit, top_index = jax.lax.top_k(ext_logits.astype(jnp.float32), 1)
    topk_p = jnp.exp(top_logit - lse)

    host_outputs = (predict, accept_length, verified_flat, next_verified_id, topk_p,
                    top_index.astype(jnp.int32), next_hidden, new_seq_lens)
    return tgt_updates, dr_updates, host_outputs


class ChainFusedRound:
    """Owns the fused round JIT and its host-side preparation."""

    def __init__(self, spec_worker):
        self.sw = spec_worker
        self.T = spec_worker.speculative_num_draft_tokens
        self.steps = spec_worker.speculative_num_steps
        self.mesh = spec_worker.mesh
        self._key = jax.random.PRNGKey(spec_worker.server_args.random_seed or 0)
        self._round_id = 0
        tgt_backend = spec_worker.target_worker.model_runner.attn_backend
        compiler_options = dict(getattr(tgt_backend, "compiler_options", None) or {})
        if jax.default_backend() == "tpu" and get_bool_env_var(
            "SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD"
        ):
            # Every forward in the round is decode-shaped (see aot_dispatch).
            compiler_options.update({
                "xla_tpu_offload_gather_to_sparsecore": "false",
                "xla_tpu_offload_all_supported_gathers_to_sparsecore": "false",
            })
        compiler_options = compiler_options or None
        # Backends enter the JIT as structure only (empty metadata): per-round metadata is
        # rebuilt in-JIT from the packed upload, so stale host metadata never reaches the
        # signature. Treedefs carry unhashable aux (dicts), so they are bound, not static.
        tgt_def = _empty_backend_treedef(spec_worker.target_worker.model_runner.attn_backend)
        dr_def = _empty_backend_treedef(spec_worker.draft_worker.draft_model_runner.attn_backend)
        self._jit = jax.jit(
            partial(_fused_round, tgt_backend_def=tgt_def, dr_backend_def=dr_def),
            donate_argnames=("tgt_pools", "dr_pools"),
            static_argnames=("tgt_state_def", "dr_state_def", "layout", "steps", "T", "bs",
                             "threshold_single", "threshold_acc", "mesh"),
            compiler_options=compiler_options,
        )

    def supports_batch(self, mwb) -> bool:
        return not (mwb.return_logprob or getattr(mwb, "return_output_logprob_only", False))

    def run(self, mwb, cur_allocate_lens):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        sw = self.sw
        dw = sw.draft_worker
        T, steps = self.T, self.steps
        tmr = sw.target_worker.model_runner
        dmr = dw.draft_model_runner

        # Host: pad to the bs bucket, build every phase's metadata, upload once.
        dw.padding_for_decode(mwb)
        spec = mwb.spec_info_padded
        seq_lens = np.asarray(mwb.seq_lens, dtype=np.int32)
        bs = len(seq_lens)
        sel = np.asarray(mwb.logits_indices_selector)
        dr_mla = _mla_backend(dmr.attn_backend)
        tgt_mla = _mla_backend(tmr.attn_backend)

        step_arrays = dr_mla.eagle_multi_step_metadata_arrays(mwb)[: steps - 1]
        verify_view = _BatchView(mwb, forward_mode=ForwardMode.TARGET_VERIFY,
                                 seq_lens=np.where(seq_lens > 0, seq_lens - 1, 0),
                                 spec_info_padded=_SpecView(draft_token_num=T,
                                                            allocate_lens=cur_allocate_lens))
        verify_arrays = tgt_mla.eagle_forward_metadata_arrays(verify_view)
        extend_view = _BatchView(mwb, forward_mode=ForwardMode.DRAFT_EXTEND,
                                 seq_lens=np.where(seq_lens > 0, seq_lens - 1 + T - 1, 0),
                                 input_ids=np.empty(bs * T, np.int32),
                                 spec_info_padded=_SpecView(allocate_lens=cur_allocate_lens))
        extend_arrays = dr_mla.eagle_forward_metadata_arrays(extend_view)

        valid = seq_lens > 0
        extend_seq_lens = np.where(valid, T, 0).astype(np.int32)
        extend_logits_indices = (np.arange(bs, dtype=np.int32) * T + T - 1).astype(np.int32)
        recurrent_indices = mwb.recurrent_indices
        if recurrent_indices is None:
            recurrent_indices = np.zeros(bs, np.int32)

        sampling_info = mwb.sampling_info
        is_greedy = bool(sampling_info.is_all_greedy)
        temperatures = _pad_rows(np.asarray(sampling_info.temperatures, np.float32).reshape(-1),
                                 bs, 1.0)
        top_ks = _pad_rows(np.asarray(sampling_info.top_ks, np.int32).reshape(-1), bs, 1)
        top_ps = _pad_rows(np.asarray(sampling_info.top_ps, np.float32).reshape(-1), bs, 1.0)
        use_top_p = bool(np.any(top_ps[sel] < 1.0))

        verified_id = np.asarray(spec.verified_id, np.int32).reshape(-1)[:bs]
        first_draft = np.asarray(spec.topk_index, np.int32).reshape(bs, -1)[:, 0]
        hidden = spec.hidden_states

        rep = NamedSharding(self.mesh, P())
        self._round_id += 1
        named_int = []
        for i, arrs in enumerate(step_arrays):
            named_int += _md_named(f"step{i}_", arrs)
        named_int += _md_named("verify_", verify_arrays) + _md_named("extend_", extend_arrays)
        named_int += [
            ("kda_cu_q", np.arange(bs + 1, dtype=np.int32) * T),  # unused by the chain kernel
            ("recurrent_indices", np.asarray(recurrent_indices, np.int32)),
            ("has_initial_state", np.ones(bs, np.int32)),
            ("extend_seq_lens", extend_seq_lens),
            ("extend_logits_indices", extend_logits_indices),
            ("seq_lens", seq_lens),
            ("verified_id", verified_id),
            ("first_draft", first_draft),
            ("top_ks", top_ks),
            ("is_greedy", np.asarray([int(is_greedy)], np.int32)),
            ("use_top_p", np.asarray([int(use_top_p)], np.int32)),
            ("round_id", np.asarray([self._round_id], np.int32)),
        ]
        named_f32 = [("temperatures", temperatures), ("top_ps", top_ps)]
        int_blob, int_layout = _pack(named_int, np.int32)
        f32_blob, f32_layout = _pack(named_f32, np.float32)
        layout = {**int_layout, **f32_layout}
        layout_key = tuple(sorted(layout.items()))
        if not isinstance(hidden, jax.Array):
            hidden = np.asarray(hidden)
        int_dev, f32_dev, hidden = jax.device_put((int_blob, f32_blob, hidden), rep)

        with jax.set_mesh(self.mesh):
            tgt_updates, dr_updates, outs = self._jit(
                tmr._model_def, tmr._model_state_def, tmr.model_state_leaves, tmr.memory_pools,
                dmr._model_def, dmr._model_state_def, dmr.model_state_leaves, dmr.memory_pools,
                get_global_expert_location_metadata(), int_dev, f32_dev, hidden, self._key,
                layout=_Layout(layout_key), steps=steps, T=T, bs=bs,
                threshold_single=float(
                    global_server_args_dict["speculative_accept_threshold_single"]),
                threshold_acc=float(global_server_args_dict["speculative_accept_threshold_acc"]),
                mesh=self.mesh,
            )
        tmr.memory_pools.replace_all(tgt_updates)
        dmr.memory_pools.replace_all(dr_updates)

        (predict, accept_length, verified_flat, next_verified, topk_p, topk_index, next_hidden,
         new_seq_lens) = jax.device_get(outs)
        real = sel
        next_draft_input = EagleDraftInput(
            verified_id=next_verified[real],
            new_seq_lens=new_seq_lens,
            allocate_lens=np.asarray(cur_allocate_lens)[: mwb.real_bs],
            hidden_states=next_hidden[real],
            topk_p=topk_p[real],
            topk_index=topk_index[real],
        )
        mwb.spec_info_padded = next_draft_input
        batch_output = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None, hidden_states=None),
            next_token_ids=predict,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            bid=mwb.bid,
            cache_miss_count=0,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )
        launch_done = getattr(mwb, "launch_done", None)
        if launch_done is not None:
            launch_done.set()
        return batch_output


class _Layout:
    """Static, hashable blob layout: name -> (offset, shape)."""

    def __init__(self, items):
        self._items = items
        self._d = dict(items)

    def __getitem__(self, name):
        return self._d[name]

    def __contains__(self, name):
        return name in self._d

    def __hash__(self):
        return hash(self._items)

    def __eq__(self, other):
        return isinstance(other, _Layout) and self._items == other._items


_MD_KEYS = ("cu_q", "cu_kv", "pi", "seq", "dist", "req_seq", "req_pi", "req_cuq", "req_dist")


def _md_named(prefix, arrays):
    return [(prefix + k, a) for k, a in zip(_MD_KEYS, arrays)]


def _pack(named, dtype):
    """Concatenate host arrays into one flat buffer; layout maps name -> (offset, shape)."""
    layout, parts, off = {}, [], 0
    for name, arr in named:
        arr = np.asarray(arr).astype(dtype, copy=False)
        layout[name] = (off, tuple(arr.shape))
        parts.append(arr.reshape(-1))
        off += arr.size
    return (np.concatenate(parts) if parts else np.zeros((0,), dtype)), layout


def _empty_backend_treedef(backend):
    empty = (HybridLinearAttnBackendMetadata()
             if isinstance(backend, HybridLinearAttnBackend) else type(backend.forward_metadata)())
    saved = backend.forward_metadata
    backend.forward_metadata = empty
    try:
        leaves, treedef = jax.tree_util.tree_flatten(backend)
    finally:
        backend.forward_metadata = saved
    assert not leaves, f"backend carries array state besides metadata: {len(leaves)} leaves"
    return treedef


class _BatchView:
    """Attribute overlay over a ModelWorkerBatch for metadata builders."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


class _SpecView:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _pad_rows(arr, n, fill):
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), fill, dtype=arr.dtype)])
