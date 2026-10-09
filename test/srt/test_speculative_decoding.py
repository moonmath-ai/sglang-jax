import os
import unittest
from types import SimpleNamespace

import requests
from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_32B,
    QWEN3_32B_EAGLE3,
    CustomTestCase,
    popen_launch_server,
)


class TestSpeculativeDecoding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            # FIXME(#1053 P1-2): spec precompile shapes don't yet fully cover
            # runtime padding (draft prefill bs<min_bucket); re-enable once
            # the BaseSpecWorker refactor unifies precompile bucket selection.
            check_cache_miss=False,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--mem-fraction-static",
                "0.8",
                "--download-dir",
                "/dev/shm",
                "--max-running-requests",
                "64",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "4096",
                "--context-length",
                "4096",
                "--speculative-draft-model-path",
                QWEN3_32B_EAGLE3,
                "--speculative-draft-model-revision",
                "67caf31f9062d7ab64872e0a111d499bc16cd205",  # this model revision has .safetensor model file, which is converted by huggingface official
                # FIXME(pc) topk > 1 has poor performance now, change it when build_tree_mask_for_draft_decode kernel is  implemented
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-steps",
                "3",
                "--speculative-num-draft-tokens",
                "4",
                # FIXME(pc) currently, spec decode is not fully compatible with scheduler overlap, rm this when fix it
                "--disable-overlap-schedule",
                "--speculative-algorithm",
                "EAGLE3",
                "--page-size",
                "64",
                "--attention-backend",
                "fa",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            # TODO(#1053 P1-2): restore num_examples=512/threads=64 once spec
            # precompile shapes fully cover the runtime buckets (currently each
            # new prefill token-count recompiles, making 512 too slow for CI).
            num_examples=64,
            num_threads=16,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.45)


@unittest.skipUnless(
    os.getenv("SGLANG_NEXTN_E2E_URL"),
    "MiMo-V2.5-Pro NEXTN needs a v6e-64 multi-host server (1T model); set "
    "SGLANG_NEXTN_E2E_URL to a running server's base URL to enable.",
)
class TestNextNV25Pro(CustomTestCase):
    """#1053 P1-8: V2.5-Pro 3-layer MTP E2E.

    Unlike ``TestSpeculativeDecoding`` this does NOT launch its own server:
    V2.5-Pro requires v6e-64 (16 hosts) which ``popen_launch_server`` can't
    orchestrate. Point ``SGLANG_NEXTN_E2E_URL`` at an externally-managed
    server started with::

        --speculative-algorithm NEXTN --speculative-num-steps 3
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
        --tp-size 64 --ep-size 64 --moe-backend epmoe
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ["SGLANG_NEXTN_E2E_URL"]
        cls.model = os.getenv("SGLANG_NEXTN_E2E_MODEL", "mimo-v2.5-pro")
        r = requests.get(f"{cls.base_url}/health", timeout=30)
        r.raise_for_status()

    def _generate(self, text: str, max_new_tokens: int = 64) -> dict:
        r = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()

    def test_greedy_sanity(self):
        d = self._generate("The capital of France is", 32)
        self.assertGreater(d["meta_info"]["completion_tokens"], 16)
        self.assertGreater(len(d["text"]), 0)

    def test_multi_prompt_stable(self):
        # Regression for the spec-decode KV leak (idle check_memory crash after
        # each finished request) and the page>=256 _fetch_mask Mosaic crash:
        # serially run mixed-length requests crossing the 256-token page
        # boundary and assert the server stays up across all of them.
        for n in (32, 280, 48, 96):
            d = self._generate("def fibonacci(n):\n    if n <= 1:\n        return n\n    return", n)
            self.assertGreaterEqual(d["meta_info"]["completion_tokens"], n)
        requests.get(f"{self.base_url}/health", timeout=10).raise_for_status()

    def test_raw_completion_accuracy(self):
        # run_eval's mmlu uses /v1/chat/completions; V2.5-Pro then emits
        # <think>...</think> reasoning which the mmlu parser can't score.
        # Use raw /generate (no chat template) so the model answers in
        # few-shot completion style. bs=1 only (Phase-1 deliverable scope).
        cases = [
            ("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"], "B"),
            (
                "Which planet is known as the Red Planet?",
                ["Venus", "Mars", "Jupiter", "Saturn"],
                "B",
            ),
            ("What is the chemical symbol for water?", ["CO2", "H2O", "NaCl", "O2"], "B"),
            ("Who wrote 'Romeo and Juliet'?", ["Dickens", "Shakespeare", "Twain", "Austen"], "B"),
            ("What is 7 * 8?", ["54", "56", "58", "64"], "B"),
        ]
        correct = 0
        for q, opts, expect in cases:
            prompt = (
                f"Question: {q}\n"
                + "\n".join(f"{c}. {o}" for c, o in zip("ABCD", opts))
                + "\nAnswer:"
            )
            out = self._generate(prompt, 4)["text"].strip()
            if out[:1].upper() == expect:
                correct += 1
        self.assertGreaterEqual(correct, 4, f"raw-completion accuracy {correct}/5")


@unittest.skipUnless(
    os.getenv("SGLANG_GLM53_FLASH_E2E") == "1",
    "GLM-5.3-Flash NEXTN needs a TPU v7x-8 (tp8/ep8) with the zai-org/GLM-5.3-Flash weights; "
    "set SGLANG_GLM53_FLASH_E2E=1 to enable.",
)
class TestNextNGLM53Flash(CustomTestCase):
    """GLM-5.3-Flash NextN MTP (one-JIT chain round, DSA + KDA) in its serving setup:
    overlap schedule and the recurrent radix cache with the extra buffer.

    Checks accuracy (gsm8k), draft quality (mean accept length), token-level agreement with
    the target's own greedy choice (teacher-forced), and long-context retrieval (needle),
    which exercises multi-page request page tables and the indexer's long buckets.
    """

    MODEL = "zai-org/GLM-5.3-Flash"

    @classmethod
    def setUpClass(cls):
        cls.model = cls.MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            device="tpu",
            check_cache_miss=False,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "8",
                "--ep-size",
                "8",
                "--attention-backend",
                "dsa_sparse",
                "--moe-backend",
                "fused_v2",
                "--page-size",
                "128",
                "--context-length",
                "16384",
                "--chunked-prefill-size",
                "4096",
                "--mem-fraction-static",
                "0.8",
                "--max-running-requests",
                "16",
                "--enable-unified-radix-tree",
                "--enable-recurrent-extra-buffer",
                "--reasoning-parser",
                "glm53",
                "--precompile-token-paddings",
                "512",
                "1024",
                "2048",
                "4096",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-draft-model-path",
                cls.MODEL,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=body, timeout=3600)
        r.raise_for_status()
        return r.json()

    def test_a_gsm8k_and_accept_length(self):
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=200,
                num_threads=16,
                max_tokens=4096,
            )
        )
        # Measured 0.985-0.99; a broken chain (e.g. a scrambled conv window) lands ~0.94.
        self.assertGreaterEqual(metrics["score"], 0.97)
        info = requests.get(f"{self.base_url}/get_server_info", timeout=30).json()
        accept_length = info["internal_states"][0]["avg_spec_accept_length"]
        # Measured ~3.6-3.7 (of 4).
        self.assertGreaterEqual(accept_length, 3.3)

    def test_b_teacher_forced_greedy(self):
        """Every token the spec round emits should be the target's own greedy next token
        (one-token prefill of the same prefix). Numerics differ between the two paths, so a
        few near-tie flips are expected (measured 1-2 of 72)."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        questions = [
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?",
            "Janet has 3 apples and buys 5 more. She gives away 2. How many apples does she have?",
            "Write one sentence about the history of the Roman Empire.",
        ]
        total = mismatches = 0
        for q in questions:
            ids = tok.encode(
                tok.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                add_special_tokens=False,
            )
            greedy = {"temperature": 0}
            out = self._post(
                "/generate",
                {"input_ids": ids, "sampling_params": greedy | {"max_new_tokens": 24}},
            )["output_ids"]
            for k, token in enumerate(out):
                ref = self._post(
                    "/generate",
                    {
                        "input_ids": ids + out[:k],
                        "sampling_params": greedy | {"max_new_tokens": 1},
                    },
                )["output_ids"][0]
                total += 1
                mismatches += ref != token
        self.assertLessEqual(mismatches, 4, f"{mismatches}/{total} teacher-forced mismatches")

    def test_c_needle(self):
        """A passcode hidden in ~4K and ~14K tokens of filler is retrieved at every depth."""
        import random

        rng = random.Random(0)
        words = ["river", "granite", "lantern", "orchard", "copper", "harbor", "meadow", "cedar"]
        failures = []
        for n_notes in (180, 650):  # ~4K and ~14.5K prompt tokens
            for depth in (0.1, 0.5, 0.9):
                notes = [
                    f"Note {i}: the {' '.join(rng.sample(words, 2))} was moved next to the "
                    f"{' '.join(rng.sample(words, 2))} on day {rng.randint(1, 28)}."
                    for i in range(n_notes)
                ]
                code = str(rng.randint(10000, 99999))
                notes.insert(int(n_notes * depth), f"Important: the secret passcode is {code}.")
                prompt = (
                    "Here is a long list of notes.\n\n"
                    + "\n".join(notes)
                    + "\n\nWhat is the secret passcode? Answer with the number only."
                )
                r = self._post(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "temperature": 0,
                        "max_tokens": 1500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if code not in (r["choices"][0]["message"].get("content") or ""):
                    failures.append((r["usage"]["prompt_tokens"], depth))
        self.assertEqual(failures, [], f"needle missed at (prompt_tokens, depth): {failures}")


if __name__ == "__main__":
    unittest.main()
