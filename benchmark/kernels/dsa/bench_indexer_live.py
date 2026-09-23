"""Real-shape indexer-only benchmark; no model or weights.

Before is the unchanged streamindex_topk_ref. Page tables retain the serving
capacity even for short live contexts, use packed token-unit offsets, and map
to disjoint physical pages in a BF16 cache. Device timings exclude compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.dsa.ref import streamindex_topk_ref

PAGE, HEADS, DIM, K, CAP_PAGES = 128, 32, 128, 2048, 7635
CASES = [
    ("b4_650", [650] * 4),
    ("b16_650", [650] * 16),
    ("b4_8k", [8192] * 4),
    ("b16_8k", [8192] * 16),
    ("b4_32k", [32768] * 4),
    ("b16_32k", [32768] * 16),
    ("b4_128k", [131072] * 4),
    ("b4_mixed", [2049, 8193, 32769, 131071]),
    ("b16_mixed", [2049, 8193, 32769, 131071] * 4),
    ("b1_256k", [262144]),
    ("b1_512k", [524288]),
    ("b1_capacity", [CAP_PAGES * PAGE]),
]


def make_host_inputs(lengths, cache):
    b = len(lengths)
    rng = np.random.default_rng(1234 + b)
    pages = np.asarray([(n + PAGE - 1) // PAGE for n in lengths], np.int32)
    assert pages.sum() <= CAP_PAGES
    physical = np.random.default_rng(5678).permutation(CAP_PAGES).astype(np.int32)
    pi = np.full(b * CAP_PAGES, CAP_PAGES, np.int32)
    pi[: pages.sum()] = physical[: pages.sum()]
    return (
        rng.normal(0, 0.2, (b, HEADS, DIM)).astype(ml_dtypes.bfloat16),
        rng.normal(0, 0.2, (b, HEADS)).astype(ml_dtypes.bfloat16),
        cache,
        np.asarray(lengths, np.int32),
        pi,
        np.arange(b + 1, dtype=np.int32),
        np.r_[0, np.cumsum(pages, dtype=np.int32) * PAGE].astype(np.int32),
        np.asarray([b, b, b], np.int32),
    )


def exact_scores(host):
    """Independent FP32 NumPy oracle, gathering only each request's live keys."""
    q, weights, cache, lengths, pages, _, offsets, _ = host
    result = []
    for row, length in enumerate(lengths):
        start = offsets[row] // PAGE
        n_pages = (int(length) + PAGE - 1) // PAGE
        keys = cache[pages[start : start + n_pages]].reshape(-1, DIM)[:length].astype(np.float32)
        dots = np.asarray(q[row], np.float32) @ keys.T
        result.append(
            (np.maximum(dots, 0) * np.asarray(weights[row], np.float32)[:, None]).sum(axis=0)
        )
    return result


def check_selection(result, scores, *, exact):
    recalls, losses = [], []
    for selected, row_scores in zip(result, scores, strict=True):
        n = min(K, len(row_scores))
        valid = selected[selected >= 0]
        if exact:
            assert len(valid) == n
        assert 0 < len(valid) <= n
        assert np.all(selected[len(valid) :] == -1)
        assert len(np.unique(valid)) == len(valid) and np.all(valid < len(row_scores))
        oracle = np.argpartition(-row_scores, n - 1)[:n]
        recalls.append(len(np.intersect1d(oracle, valid)) / n)
        loss = max(0.0, float(row_scores[oracle].min() - row_scores[valid].min()))
        losses.append(loss)
        if exact:
            # Accommodate FP32 reduction order around ties, never a gross rank error.
            assert loss <= 2e-5 * max(1.0, float(np.abs(row_scores).max())), (loss, recalls[-1])
    return {"exact_set_recall_per_row": recalls, "max_cutoff_score_loss_per_row": losses}


def compile_fn(variant, b, inputs):
    if variant == "before":
        fn = streamindex_topk_ref

        def run(*xs):
            return fn(*xs, k=K, pages_per_seq=CAP_PAGES, one_token_per_seq=True)

    else:
        from sgl_jax.srt.kernels.dsa.streamindex_live import streamindex_topk_live

        def run(*xs):
            return streamindex_topk_live(*xs, k=K, pages_per_seq=CAP_PAGES)

    run.__name__ = run.__qualname__ = f"indexer_{variant}_b{b}"
    return jax.jit(run).lower(*inputs).compile()


def device_times(path, order):
    from xprof import profile_data

    events = []
    for source in path.rglob("*.xplane.pb"):
        with profile_data.ProfileData.from_file(str(source)) as pd:
            for plane in pd.planes:
                if plane.name != "/device:TPU:0":
                    continue
                for line in plane.lines:
                    if line.name != "XLA Modules":
                        continue
                    for e in line.events:
                        name = e.name.split("(", 1)[0]
                        if name.startswith("jit_indexer_"):
                            events.append((e.start_ns, name, e.duration_ns / 1000))
    events.sort()
    assert len(events) == len(order), (len(events), len(order))
    result = {}
    for (_, name, us), (variant, case, b) in zip(events, order, strict=True):
        assert name == f"jit_indexer_{variant}_b{b}", name
        result.setdefault((variant, case), []).append(us)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["before", "after", "both"], default="before")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    assert device.platform == "tpu"
    print("Device", device, "kind", device.device_kind, "devices", len(jax.devices()), flush=True)
    cache = (
        np.random.default_rng(4321)
        .normal(0, 0.2, (CAP_PAGES + 1, PAGE, DIM))
        .astype(ml_dtypes.bfloat16)
    )
    cache[-1] = 0
    cache_hash = hashlib.sha256(cache.tobytes()).hexdigest()
    device_cache = jax.device_put(cache, device)
    variants = ["before", "after"] if args.variant == "both" else [args.variant]
    executables = {}
    jobs = []
    rows = []
    cases = [x for x in CASES if not args.cases or x[0] in args.cases]
    for case, lengths in cases:
        host = make_host_inputs(lengths, cache)
        inputs = tuple(
            device_cache if i == 2 else jax.device_put(x, device) for i, x in enumerate(host)
        )
        oracle_scores = exact_scores(host)
        for variant in variants:
            key = (variant, len(lengths))
            if key not in executables:
                print("Compiling", key, flush=True)
                start = time.monotonic()
                executables[key] = compile_fn(variant, len(lengths), inputs)
                print("Compiled", key, "seconds", round(time.monotonic() - start, 2), flush=True)
                (
                    args.output.parent / f"{args.output.stem}_{variant}_b{len(lengths)}.hlo.txt"
                ).write_text(executables[key].as_text())
            fn = executables[key]
            result = np.asarray(fn(*inputs))
            for r, n in zip(result, lengths, strict=True):
                valid = r[r >= 0]
                assert np.all(valid < n) and len(np.unique(valid)) == len(valid)
            correctness = check_selection(result, oracle_scores, exact=variant == "after")
            np.save(args.output.parent / f"{args.output.stem}_{variant}_{case}_indices.npy", result)
            for _ in range(3):
                jax.block_until_ready(fn(*inputs))
            walls = []
            for _ in range(10):
                start = time.perf_counter_ns()
                jax.block_until_ready(fn(*inputs))
                walls.append((time.perf_counter_ns() - start) / 1000)
            rows.append(
                {
                    "case": case,
                    "variant": variant,
                    "batch": len(lengths),
                    "live_lengths": lengths,
                    "wall_us": walls,
                    "wall_median_us": statistics.median(walls),
                    "valid_indices_per_row": (result >= 0).sum(axis=1).tolist(),
                    "correctness": correctness,
                }
            )
            jobs.append((variant, case, len(lengths), fn, inputs))
            print(
                "Warmed", variant, case, "wall_us", round(statistics.median(walls), 2), flush=True
            )
    profile = args.output.parent / (args.output.stem + "_profile")
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    order = []
    print("Capturing warmed device times", flush=True)
    with jax.profiler.trace(str(profile), profiler_options=options):
        for repeat in range(args.repeats):
            for variant, case, b, fn, inputs in jobs if repeat % 2 == 0 else reversed(jobs):
                order.append((variant, case, b))
                jax.block_until_ready(fn(*inputs))
    timings = device_times(profile, order)
    for row in rows:
        ds = timings[row["variant"], row["case"]]
        assert len(ds) == args.repeats
        row["device_us"] = ds
        row["device_median_us"] = statistics.median(ds)
        print(
            row["variant"], row["case"], "device_us", round(row["device_median_us"], 3), flush=True
        )
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "device_kind": device.device_kind,
        "versions": {p: importlib.metadata.version(p) for p in ["jax", "jaxlib", "libtpu"]},
        "parameters": {
            "page_size": PAGE,
            "heads": HEADS,
            "dim": DIM,
            "k": K,
            "pages_per_seq": CAP_PAGES,
            "cache_shape": list(cache.shape),
            "cache_dtype": "bfloat16",
            "cache_sha256": cache_hash,
            "q_and_weights_dtype": "bfloat16",
            "device_repeats": args.repeats,
        },
        "method": "One TPU; replicated per-device serving shape; disjoint live pages; packed offsets in token units; randomized BF16 data. XLA Modules durations exclude compile/warmup/host overhead.",
        "profile_dir": str(profile),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
