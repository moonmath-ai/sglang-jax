"""Compare the original one-hot page deduplication with compute_topk_pages.

No model or weights are loaded. Inputs live on one TPU device, matching the
per-device shape of the indexer's tensor-parallel replicated page selection.
Compilation and warmup are excluded. Report synchronized wall times separately
from device module durations read from a short XPlane capture (requires xprof).

From the repository root:
  PYTHONPATH=python python benchmark/kernels/dsa/bench_page_dedup.py \
    --output /tmp/page_dedup/results.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.sparse_mla import compute_topk_pages


def one_hot_before(topk_indices, *, page_size, pages_per_seq, k_pages_max):
    """Original compute_topk_pages from 6fb37c5, retained as the A/B baseline."""
    valid = topk_indices >= 0
    page_local = jnp.where(valid, topk_indices // page_size, pages_per_seq)
    page_hits = jax.nn.one_hot(page_local, pages_per_seq, dtype=jnp.int32)
    page_mask = jnp.any(page_hits, axis=1)
    n_hit = jnp.sum(page_mask, axis=-1)
    k_eff = min(k_pages_max, pages_per_seq)
    _, hit_pages = jax.lax.top_k(page_mask.astype(jnp.int32), k_eff)
    hit_pages = jnp.pad(hit_pages, ((0, 0), (0, k_pages_max - k_eff)))
    hit_valid = jnp.arange(k_pages_max)[None, :] < n_hit[:, None]
    return jnp.where(hit_valid, hit_pages, -1)


def make_input(batch, selected, page_size, pages_per_seq, pattern):
    rng = np.random.default_rng(42)
    tokens = np.full((batch, selected), -1, np.int32)
    for row in range(batch):
        context = min(512 + 128 * (row % 4), pages_per_seq * page_size)
        if pattern == "spread":
            context = pages_per_seq * page_size
        n_valid = min(selected, context)
        tokens[row, :n_valid] = rng.choice(context, n_valid, replace=False)
    return tokens


def expected_pages(tokens, *, page_size, pages_per_seq, k_pages_max):
    out = np.full((tokens.shape[0], k_pages_max), -1, np.int32)
    for row, ids in enumerate(tokens):
        valid = ids[(ids >= 0) & (ids // page_size < pages_per_seq)]
        pages = np.unique(valid // page_size)[:k_pages_max]
        out[row, : len(pages)] = pages
    return out


def compile_variant(fn, label, x, kwargs):
    def run(tokens):
        return fn(tokens, **kwargs)

    run.__name__ = run.__qualname__ = label
    return jax.jit(run).lower(x).compile()


def read_device_times(profile_dir, execution_order):
    from xprof import profile_data

    events = []
    paths = list(profile_dir.glob("plugins/profile/**/*.xplane.pb"))
    if not paths:
        raise RuntimeError(f"No raw XPlane capture under {profile_dir}")
    for path in paths:
        with profile_data.ProfileData.from_file(str(path)) as pd:
            for plane in pd.planes:
                if plane.name != "/device:TPU:0":
                    continue
                for line in plane.lines:
                    if line.name != "XLA Modules":
                        continue
                    for event in line.events:
                        name = event.name.split("(", 1)[0]
                        if name.startswith("jit_pages_"):
                            events.append((event.start_ns, name, event.duration_ns / 1000))
    events.sort()
    if len(events) != len(execution_order):
        raise RuntimeError(f"Incomplete device trace: {len(events)}/{len(execution_order)} calls")
    timings = {}
    for (_, program, duration_us), label in zip(events, execution_order, strict=True):
        # Both data patterns use the same executable. Attribute by the recorded
        # invocation order rather than assuming compilation preserves case names.
        expected_program = "jit_" + label.rsplit("_", 1)[0]
        if program != expected_program:
            raise RuntimeError(f"Unexpected device program: {program}, wanted {expected_program}")
        timings.setdefault(label, []).append(duration_us)
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--selected", type=int, default=2048)
    parser.add_argument("--pages-per-seq", type=int, default=7635)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--wall-repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.batches + [args.repeats, args.wall_repeats]) < 1:
        parser.error("Batch sizes and iteration counts must be positive")
    if args.output.exists():
        parser.error("Use a new output path to preserve previous benchmark results")
    device = jax.devices()[0]
    if device.platform != "tpu":
        raise RuntimeError(f"Device timing benchmark requires TPU, got {device}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile_dir = args.output.parent / (args.output.stem + "_profile")
    if profile_dir.exists():
        parser.error("Use a new output path; a profile already exists for this name")
    kwargs = dict(
        page_size=args.page_size, pages_per_seq=args.pages_per_seq, k_pages_max=args.budget
    )
    print(f"Device: {device}; kind={device.device_kind}; total devices={len(jax.devices())}")
    print(f"Input: int32[B,{args.selected}], parameters={kwargs}", flush=True)
    jobs = []
    rows = []
    executables = {}
    for pattern in ("short", "spread"):
        for batch in args.batches:
            host_input = make_input(
                batch, args.selected, args.page_size, args.pages_per_seq, pattern
            )
            x = jax.device_put(host_input, device)
            expected = expected_pages(host_input, **kwargs)
            variants = []
            for variant, fn in (("before", one_hot_before), ("after", compute_topk_pages)):
                label = f"pages_{variant}_b{batch}_{pattern}"
                key = (variant, batch)
                if key not in executables:
                    executables[key] = compile_variant(fn, f"pages_{variant}_b{batch}", x, kwargs)
                compiled = executables[key]
                np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
                for _ in range(5):
                    jax.block_until_ready(compiled(x))
                if batch == 4 and pattern == "short":
                    (args.output.parent / f"{args.output.stem}_{variant}_b4.hlo.txt").write_text(
                        compiled.as_text()
                    )
                variants.append((variant, label, compiled))
                jobs.append((label, compiled, x))
            wall_samples = {"before": [], "after": []}
            for repeat in range(args.wall_repeats):
                order = variants if repeat % 2 == 0 else reversed(variants)
                for variant, _, compiled in order:
                    start = time.perf_counter_ns()
                    jax.block_until_ready(compiled(x))
                    wall_samples[variant].append((time.perf_counter_ns() - start) / 1000)
            rows.append(
                {
                    "batch": batch,
                    "pattern": pattern,
                    "parity": "exact_numpy_unique",
                    "wall_us": wall_samples,
                }
            )
            print(f"Compiled and verified: B={batch}, pattern={pattern}", flush=True)

    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 1
    options.python_tracer_level = 0
    print("Capturing device timings for warmed executables...", flush=True)
    execution_order = []
    with jax.profiler.trace(str(profile_dir), profiler_options=options):
        # Alternate before/after and reverse order each round to reduce order bias.
        for repeat in range(args.repeats):
            order = jobs if repeat % 2 == 0 else reversed(jobs)
            for label, compiled, x in order:
                execution_order.append(label)
                jax.block_until_ready(compiled(x))
    timings = read_device_times(profile_dir, execution_order)
    print("pattern  batch   before_us   after_us   speedup", flush=True)
    for row in rows:
        samples = {}
        for variant in ("before", "after"):
            name = f"pages_{variant}_b{row['batch']}_{row['pattern']}"
            ds = timings.get(name, [])
            if len(ds) != args.repeats:
                raise RuntimeError(f"Incomplete device trace: {name}, {len(ds)}/{args.repeats}")
            samples[variant] = ds
        row["device_us"] = samples
        row["device_median_us"] = {v: statistics.median(ds) for v, ds in samples.items()}
        row["speedup"] = row["device_median_us"]["before"] / row["device_median_us"]["after"]
        print(
            f"{row['pattern']:>7} {row['batch']:>6} "
            f"{row['device_median_us']['before']:>11.3f} "
            f"{row['device_median_us']['after']:>10.3f} {row['speedup']:>9.2f}x",
            flush=True,
        )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_kind": device.device_kind,
        "execution": "single TPU device; TPU:0 XLA Modules duration; no model",
        "versions": {
            package: importlib.metadata.version(package) for package in ("jax", "jaxlib", "libtpu")
        },
        "parameters": {**kwargs, "selected": args.selected, "device_repeats": args.repeats},
        "profile_dir": str(profile_dir),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
