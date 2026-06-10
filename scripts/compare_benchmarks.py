#!/usr/bin/env python3
"""Compare benchmark results from multiple SAM model deployments.

Usage:
    python3 scripts/compare_benchmarks.py benchmark_results/*.json
    python3 scripts/compare_benchmarks.py results_a.json results_b.json
"""
import argparse
import json
import sys


def load_result(path):
    with open(path) as f:
        return json.load(f)


def print_comparison(results: list[tuple[str, dict]]):
    print("=" * 80)
    print("SAM MODEL BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    labels = [label for label, _ in results]
    col_width = max(20, max(len(l) for l in labels) + 2)

    def row(metric, values):
        print(f"  {metric:<30}", end="")
        for v in values:
            print(f"{str(v):>{col_width}}", end="")
        print()

    row("", labels)
    print("  " + "-" * (30 + col_width * len(labels)))

    # Model info
    row("Model", [r["model"] for _, r in results])
    row("Params", [r["model_params"] for _, r in results])
    row("Device", [r["device"] for _, r in results])
    row("Resolution", [r["resolution"] for _, r in results])
    row("Frames", [r["num_frames"] for _, r in results])
    print()

    # Init
    row("Init (ms)", [r["results"]["init"]["time_ms"] for _, r in results])
    print()

    # Clicks
    for i in range(max(len(r["results"]["clicks"]) for _, r in results)):
        times = []
        for _, r in results:
            clicks = r["results"]["clicks"]
            times.append(f"{clicks[i]['time_ms']}ms" if i < len(clicks) else "—")
        row(f"Click obj {i+1} (ms)", times)
    print()

    # Propagation
    row("Propagated frames", [r["results"]["propagation"]["total_frames"] for _, r in results])
    row("Total time (s)", [f"{r['results']['propagation']['total_time_ms']/1000:.1f}" for _, r in results])
    row("Avg ms/frame", [r["results"]["propagation"]["avg_ms_per_frame"] for _, r in results])
    row("FPS", [r["results"]["propagation"]["fps"] for _, r in results])
    print()

    # Object tracking
    for _, r in results:
        for oid, s in r["results"]["propagation"]["object_summary"].items():
            break
        break
    for oid in sorted(results[0][1]["results"]["propagation"]["object_summary"].keys()):
        lost_vals = []
        for _, r in results:
            s = r["results"]["propagation"]["object_summary"].get(oid, {})
            lost_vals.append(f"{s.get('lost_frames', '?')}/{s.get('total_frames', '?')}")
        row(f"Obj {oid} lost frames", lost_vals)
    print()

    # Memory
    row("GPU after init (MB)", [
        r["results"]["memory"]["after_init"].get("gpu_mb", "N/A") for _, r in results
    ])
    row("GPU after prop (MB)", [
        r["results"]["memory"]["after_propagation"].get("gpu_mb", "N/A") for _, r in results
    ])
    row("GPU after close (MB)", [
        r["results"]["memory"]["after_close"].get("gpu_mb", "N/A") for _, r in results
    ])
    row("RSS peak (MB)", [
        r["results"]["memory"]["after_propagation"].get("rss_mb", "N/A") for _, r in results
    ])

    # Winner
    print()
    print("=" * 80)
    fps_values = [(label, r["results"]["propagation"]["fps"]) for label, r in results]
    winner = max(fps_values, key=lambda x: x[1])
    print(f"  FASTEST: {winner[0]} at {winner[1]} FPS")

    mem_values = [(label, r["results"]["memory"]["after_propagation"].get("gpu_mb", float('inf')))
                  for label, r in results]
    mem_winner = min(mem_values, key=lambda x: x[1])
    if mem_winner[1] != float('inf'):
        print(f"  LEAST GPU: {mem_winner[0]} at {mem_winner[1]} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare SAM benchmark results")
    parser.add_argument("files", nargs="*", help="Benchmark JSON files to compare")
    args = parser.parse_args()

    results = []
    for f in (args.files or []):
        label = f.rsplit("/", 1)[-1].replace(".json", "")
        results.append((label, load_result(f)))

    if len(results) < 1:
        parser.error("Provide at least one benchmark file")

    print_comparison(results)


if __name__ == "__main__":
    main()
