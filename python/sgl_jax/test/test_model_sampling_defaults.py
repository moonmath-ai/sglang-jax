"""Unset OpenAI sampling fields take the model's generation_config.json values."""

import json
from types import SimpleNamespace

from sgl_jax.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sgl_jax.srt.entrypoints.openai.serving_chat import OpenAIServingChat


def _serving(tmp_path, preferred=None):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1], "temperature": 1.0, "top_p": 0.95})
    )
    s = OpenAIServingChat.__new__(OpenAIServingChat)
    s.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(model_path=str(tmp_path), preferred_sampling_params=preferred)
    )
    return s


def _params(serving, **fields):
    req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}], **fields)
    return serving._apply_model_sampling_defaults(
        req, {"temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k}
    )


def test_unset_fields_use_model_defaults(tmp_path):
    p = _params(_serving(tmp_path))
    assert p["temperature"] == 1.0 and p["top_p"] == 0.95


def test_client_values_win(tmp_path):
    p = _params(_serving(tmp_path), temperature=0.2, top_p=1.0)
    assert p["temperature"] == 0.2 and p["top_p"] == 1.0


def test_preferred_sampling_params_override(tmp_path):
    p = _params(_serving(tmp_path, preferred=json.dumps({"top_p": 0.9})))
    assert p["temperature"] == 1.0 and p["top_p"] == 0.9


def test_no_generation_config(tmp_path):
    s = OpenAIServingChat.__new__(OpenAIServingChat)
    s.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(
            model_path=str(tmp_path / "missing"), preferred_sampling_params=None
        )
    )
    req = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "hi"}])
    p = s._apply_model_sampling_defaults(req, {"temperature": req.temperature})
    assert p["temperature"] == req.temperature
