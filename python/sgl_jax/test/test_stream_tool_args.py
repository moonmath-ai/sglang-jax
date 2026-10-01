"""Final-chunk tool argument reconciliation in streaming chat (serving_chat).

With several tokens per step (speculative decoding), the last argument characters, the
closing tags and the stop can arrive in one chunk; the client-side concatenation of the
argument deltas must still equal the parsed arguments."""

import json

import pytest

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.entrypoints.openai.serving_chat import _reconcile_final_tool_args
from sgl_jax.srt.function_call.glm47_moe_detector import Glm47MoeDetector

TOOLS = [
    Tool(
        type="function",
        function={
            "name": "read",
            "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}},
        },
    )
]
PATH = "/var/folders/2h/s71cc8/T/opencode/mona_from_memory.py"
CALL = f"<tool_call>read<arg_key>filePath</arg_key><arg_value>{PATH}</arg_value></tool_call>"


def _stream(chunks, n_calls=1):
    """Feed chunks; the last one is the stop chunk. Returns the per-tool argument text."""
    d = Glm47MoeDetector()
    args = {}
    for i, chunk in enumerate(chunks):
        calls = d.parse_streaming_increment(chunk, TOOLS).calls
        if i == len(chunks) - 1 and calls:
            _reconcile_final_tool_args(d, calls)
        for c in calls:
            args[c.tool_index] = args.get(c.tool_index, "") + (c.parameters or "")
    assert len(args) == n_calls
    return [json.loads(args[k]) for k in sorted(args)]


@pytest.mark.parametrize(
    "tail", [".py</arg_value></tool_call>", "</arg_value></tool_call>", "memory.py</arg_value>"]
)
def test_last_chunk_with_closing_tags(tail):
    split = CALL.index(tail)
    chunks = [CALL[:20], CALL[20:split], CALL[split:]]
    if not CALL[split:].endswith("</tool_call>"):
        chunks[-1:] = [CALL[split:], "</tool_call>"]
    assert _stream(chunks) == [{"filePath": PATH}]


def test_whole_call_in_stop_chunk():
    assert _stream([CALL]) == [{"filePath": PATH}]


def test_second_tool_call_ends_in_stop_chunk():
    other = CALL.replace("mona_from_memory.py", "draw_face.ts")
    text = CALL + other
    split = text.index(".ts</arg_value>")
    out = _stream([text[:split], text[split:]], n_calls=2)
    assert out == [{"filePath": PATH}, {"filePath": PATH.replace("mona_from_memory.py", "draw_face.ts")}]
