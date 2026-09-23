#!/usr/bin/env python3
"""Concurrency benchmark for the GLM-5.3-Flash TPU endpoint.

Sends `--in-tok` input tokens and requests `--out-tok` output tokens, at concurrency levels
`--concurrency 1,2,4,8,16`. Measures per-request latency, TTFT (via streaming), and aggregate
decode throughput.

Usage:
  python bench_concurrency.py --url http://127.0.0.1:30333 --model /mnt/emir-disk/real \
      --in-tok 512 --out-tok 512 --concurrency 2,4
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def make_prompt(n_tok):
    # ~1.3 tokens per word for this filler; padded to roughly n_tok input tokens
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dogs", "and", "runs"]
    return " ".join(words * (n_tok // 2 + 1))[: n_tok * 6]


def one_request(url, model, prompt, out_tok, stream=True):
    """Returns (ttft_s, total_s, n_out). Streams so we can measure TTFT."""
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": out_tok,
        "temperature": 0, "stream": stream, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if ttft is None:
                ttft = time.time() - t0
            ch = d.get("choices", [{}])[0]
            if ch.get("text"):
                n += 1
    return ttft, time.time() - t0, n


def run_level(url, model, prompt, out_tok, c):
    with ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(one_request, url, model, prompt, out_tok) for _ in range(c)]
        t0 = time.time()
        res = [f.result() for f in futs]
        wall = time.time() - t0
    lat = [x[1] for x in res]
    ttfts = [x[0] for x in res if x[0] is not None]
    n_out = sum(x[2] for x in res)
    total_out = n_out
    return {
        "concurrency": c,
        "wall_s": wall,
        "ttft_avg_ms": (statistics.mean(ttfts) * 1000) if ttfts else None,
        "lat_avg_ms": statistics.mean(lat) * 1000,
        "lat_p50_ms": statistics.median(lat) * 1000,
        "lat_max_ms": max(lat) * 1000,
        "tokens_out": total_out,
        "decode_tok_s_aggregate": total_out / wall,
        "decode_tok_s_per_stream": (total_out / wall) / c,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30333/v1/completions")
    ap.add_argument("--model", default="/mnt/emir-disk/real")
    ap.add_argument("--in-tok", type=int, default=512)
    ap.add_argument("--out-tok", type=int, default=512)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    args = ap.parse_args()

    prompt = make_prompt(args.in_tok)
    levels = [int(x) for x in args.concurrency.split(",")]
    print(f"in~{args.in_tok} out={args.out_tok} tok, levels={levels}")
    print(f"{'c':>3} {'wall_s':>8} {'TTFT_ms':>9} {'lat_avg':>9} {'lat_p50':>9} {'lat_max':>9} "
          f"{'agg_tok/s':>10} {'per_stream':>11}")
    for c in levels:
        r = run_level(args.url, args.model, prompt, args.out_tok, c)
        print(f"{r['concurrency']:>3} {r['wall_s']:>8.2f} "
              f"{(r['ttft_avg_ms'] or 0):>9.0f} {r['lat_avg_ms']:>9.0f} {r['lat_p50_ms']:>9.0f} "
              f"{r['lat_max_ms']:>9.0f} {r['decode_tok_s_aggregate']:>10.1f} "
              f"{r['decode_tok_s_per_stream']:>11.1f}")


if __name__ == "__main__":
    main()