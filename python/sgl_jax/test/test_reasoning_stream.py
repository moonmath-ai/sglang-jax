"""Streaming reasoning parser: text held back as a possible tag prefix must not be lost.

With several tokens per step (speculative decoding) a chunk can be exactly "<" or "</"
(the start of </think> / <tool_call>); after reasoning ended, the next chunk used to
return only its own text and drop the held-back prefix (</artifactId> -> artifactId>).
"""

import random

import pytest

from sgl_jax.srt.reasoning_parser import ReasoningParser

TEXT = (
    "plan the edit</think>Here:\n<dependency>\n  <artifactId>api</artifactId>\n</dependency>\n"
    "a <| b </ c\n<tool_call>edit<arg_key>s</arg_key><arg_value><v>1</v>\n</p></arg_value></tool_call>"
)
# Token-like pieces, including standalone "<" and "</" as the GLM tokenizer produces them.
PIECES = ["plan", " the", " edit", "</think>", "Here", ":\n", "<", "dependency", ">\n", "  <",
          "artifactId", ">", "api", "</", "artifactId", ">\n", "</", "dependency", ">\n", "a", " <",
          "|", " b", " </", " c", "\n", "<tool_call>", "edit", "<arg_key>", "s", "</arg_key>",
          "<arg_value>", "<", "v", ">", "1", "</", "v", ">\n", "</", "p", ">", "</arg_value>",
          "</tool_call>"]


def _stream(chunks):
    p = ReasoningParser("glm45", stream_reasoning=True)
    reasoning, normal = "", ""
    for c in chunks:
        r, n = p.parse_stream_chunk(c)
        reasoning += r or ""
        normal += n or ""
    return reasoning, normal


def test_pieces_reassemble():
    assert "".join(PIECES) == TEXT


@pytest.mark.parametrize("seed", range(50))
def test_stream_matches_non_stream(seed):
    r = random.Random(seed)
    chunks, i = [], 0
    while i < len(PIECES):
        k = r.randint(1, 4)
        chunks.append("".join(PIECES[i:i + k]))
        i += k
    ref_reasoning, ref_normal = ReasoningParser("glm45").parse_non_stream(TEXT)
    reasoning, normal = _stream(chunks)
    assert normal.strip() == ref_normal.strip()
    assert reasoning.strip() == ref_reasoning.strip()


def test_single_token_chunks():
    _, normal = _stream(PIECES)
    assert "</artifactId>" in normal and "</dependency>" in normal and "</p>" in normal
