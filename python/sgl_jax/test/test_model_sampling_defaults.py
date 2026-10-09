"""Unset OpenAI sampling fields take the model's generation_config values."""

import json
from types import SimpleNamespace

from sgl_jax.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sgl_jax.srt.hf_transformers_utils import get_model_sampling_defaults

GEN_CONFIG = {"eos_token_id": [1], "temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.1}


def _args(model_path, preferred=None):
    return SimpleNamespace(
        model_path=str(model_path),
        trust_remote_code=False,
        revision=None,
        preferred_sampling_params=preferred,
    )


def _serving(tmp_path, preferred=None):
    (tmp_path / "generation_config.json").write_text(json.dumps(GEN_CONFIG))
    s = OpenAIServingChat.__new__(OpenAIServingChat)
    s.tokenizer_manager = SimpleNamespace(server_args=_args(tmp_path, preferred))
    return s


def _params(serving, **fields):
    req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}], **fields)
    return serving._apply_model_sampling_defaults(
        req, {"temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k}
    )


def test_unset_fields_use_model_defaults(tmp_path):
    p = _params(_serving(tmp_path))
    # temperature 1.0 equals the old transformers default and must still be picked up.
    assert p["temperature"] == 1.0 and p["top_p"] == 0.95
    assert "repetition_penalty" not in p


def test_client_values_win(tmp_path):
    p = _params(_serving(tmp_path), temperature=0.2, top_p=1.0)
    assert p["temperature"] == 0.2 and p["top_p"] == 1.0


def test_preferred_sampling_params_override(tmp_path):
    p = _params(_serving(tmp_path, preferred=json.dumps({"top_p": 0.9})))
    assert p["temperature"] == 1.0 and p["top_p"] == 0.9


def test_no_generation_config(tmp_path):
    s = OpenAIServingChat.__new__(OpenAIServingChat)
    s.tokenizer_manager = SimpleNamespace(server_args=_args(tmp_path / "missing"))
    req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}])
    p = s._apply_model_sampling_defaults(req, {"temperature": req.temperature})
    assert p["temperature"] == req.temperature


def test_shared_helper_used_by_engine(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(GEN_CONFIG))
    preferred = json.dumps({"max_new_tokens": 64, "top_p": 0.8, "unknown": 1})
    assert get_model_sampling_defaults(_args(tmp_path, preferred)) == {
        "repetition_penalty": 1.1,
        "temperature": 1.0,
        "top_p": 0.8,
        "max_new_tokens": 64,
    }
    assert get_model_sampling_defaults(_args(tmp_path / "missing")) == {}
