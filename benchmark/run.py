#!/usr/bin/env python3
"""
Benchmark CLI - Run performance benchmarks on the web screenshot parser.

Usage:
    python -m benchmark.run --url "https://example.com" --runs 10
    python -m benchmark.run --category short --runs 5
    python -m benchmark.run --cold-start
    python -m benchmark.run ocr --category short
    python -m benchmark.run compare --category short
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import Benchmark, BenchmarkResult, BENCHMARK_URLS
from benchmark.benchmarker import print_benchmark_results


def cmd_screenshot_benchmark(args):
    """Run screenshot + extraction benchmark."""
    benchmark = Benchmark()

    # Get URLs
    if args.category:
        urls = BENCHMARK_URLS.get(args.category, BENCHMARK_URLS["all"])
    else:
        urls = args.urls if args.urls else BENCHMARK_URLS["short"]

    print(f"\nRunning benchmark:")
    print(f"  URLs: {len(urls)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Runs per URL: {args.runs}")
    print(f"  Category: {args.category or 'custom'}")

    results = benchmark.run_screenshot_benchmark(
        urls=urls,
        warmup=args.warmup,
        runs=args.runs,
        category=args.category or "custom",
    )

    print_benchmark_results([results])
    return results


def cmd_cold_start(args):
    """Run cold start benchmark."""
    benchmark = Benchmark()

    print(f"\nRunning cold start benchmark:")
    print(f"  Runs: {args.runs}")

    result = benchmark.measure_cold_start(runs=args.runs)

    print_benchmark_results([result])
    return result


def cmd_e2e(args):
    """Run end-to-end benchmark (async)."""
    benchmark = Benchmark()

    if args.category:
        urls = BENCHMARK_URLS.get(args.category, BENCHMARK_URLS["all"])
    else:
        urls = args.urls if args.urls else BENCHMARK_URLS["short"]

    print(f"\nRunning end-to-end benchmark:")
    print(f"  URLs: {len(urls)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Runs per URL: {args.runs}")

    result = asyncio.run(
        benchmark.measure_end_to_end(
            urls=urls,
            warmup=args.warmup,
            runs=args.runs,
        )
    )

    print_benchmark_results([result])
    return result


def cmd_ocr_benchmark(args):
    """Run OCR-only benchmark."""
    benchmark = Benchmark()

    if args.category:
        urls = BENCHMARK_URLS.get(args.category, BENCHMARK_URLS["all"])
    else:
        urls = args.urls if args.urls else BENCHMARK_URLS["short"]

    print(f"\nRunning OCR-only benchmark:")
    print(f"  URLs: {len(urls)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Runs per URL: {args.runs}")

    # Pre-capture screenshots
    images = []
    for url in urls:
        try:
            img = benchmark.screenshot_capture.capture(url)
            images.append(img)
        except Exception as e:
            print(f"  Warning: Failed to capture {url}: {e}")

    if not images:
        print("No screenshots captured, aborting.")
        return None

    result = benchmark.measure_ocr_only(images, warmup=args.warmup, runs=args.runs)
    print_benchmark_results([result])
    return result


def cmd_compare(args):
    """Compare OCR vs VL vs Hybrid extraction methods."""
    benchmark = Benchmark()

    if args.category:
        urls = BENCHMARK_URLS.get(args.category, BENCHMARK_URLS["all"])
    else:
        urls = args.urls if args.urls else BENCHMARK_URLS["short"]

    print(f"\nRunning comparison benchmark:")
    print(f"  URLs: {len(urls)}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Runs per URL: {args.runs}")
    print(f"\nThis will test three methods:")
    print(f"  1. OCR-only (fast, no GPU)")
    print(f"  2. VL-only (vision model)")
    print(f"  3. Hybrid (OCR + VL fallback)")
    print()

    # Pre-capture screenshots
    images = []
    for url in urls:
        try:
            img = benchmark.screenshot_capture.capture(url)
            images.append(img)
        except Exception as e:
            print(f"  Warning: Failed to capture {url}: {e}")

    if not images:
        print("No screenshots captured, aborting.")
        return None

    results = benchmark.measure_hybrid_comparison(
        images, warmup=args.warmup, runs=args.runs
    )

    # Print comparison results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS: OCR vs VL vs Hybrid")
    print("=" * 70)

    for method, data in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Mean Time: {data['mean_ms']:.1f} ms")
        print(f"  Avg Confidence: {data['avg_confidence']:.2%}")
        if 'vl_fallbacks' in data:
            print(f"  VL Fallbacks: {data['vl_fallbacks']}/{data['timings'].__len__()}")
            print(f"  Fallback Rate: {data['fallback_rate']:.1%}")

    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  {'Method':<15} {'Time (ms)':<12} {'Confidence':<12} {'Speedup vs VL'}")
    print("-" * 70)

    vl_time = results["vl_only"]["mean_ms"]
    for method in ["ocr_only", "vl_only", "hybrid"]:
        data = results[method]
        speedup = vl_time / data["mean_ms"] if data["mean_ms"] > 0 else 0
        print(f"  {method:<15} {data['mean_ms']:<12.1f} {data['avg_confidence']:<12.2%} {speedup:.2f}x")

    print("=" * 70)
    return results


def cmd_compare_quantization(args):
    """Compare different configurations."""
    print("\nComparison mode:")
    print("  This would compare different quantization settings")
    print("  For now, run benchmarks with different settings manually")
    print("\nExample:")
    print("  # Test with no quantization")
    print("  QUANTIZATION=none python -m benchmark.run --category short")
    print()
    print("  # Test with INT4 quantization")
    print("  QUANTIZATION=int4 python -m benchmark.run --category short")


def main():
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks on web screenshot parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Benchmark commands")

    # Screenshot benchmark
    screenshot_parser = subparsers.add_parser("screenshot", help="Benchmark screenshot + extraction")
    screenshot_parser.add_argument("--urls", nargs="+", help="URLs to benchmark")
    screenshot_parser.add_argument("--category", choices=["short", "long", "complex", "all"], help="Use preset URLs")
    screenshot_parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    screenshot_parser.add_argument("--runs", type=int, default=10, help="Measured runs per URL")
    screenshot_parser.set_defaults(func=cmd_screenshot_benchmark)

    # Cold start benchmark
    cold_parser = subparsers.add_parser("cold-start", help="Benchmark cold start time")
    cold_parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    cold_parser.set_defaults(func=cmd_cold_start)

    # End-to-end benchmark
    e2e_parser = subparsers.add_parser("e2e", help="Benchmark end-to-end (async)")
    e2e_parser.add_argument("--urls", nargs="+", help="URLs to benchmark")
    e2e_parser.add_argument("--category", choices=["short", "long", "complex", "all"], help="Use preset URLs")
    e2e_parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    e2e_parser.add_argument("--runs", type=int, default=5, help="Measured runs per URL")
    e2e_parser.set_defaults(func=cmd_e2e)

    # OCR benchmark
    ocr_parser = subparsers.add_parser("ocr", help="Benchmark OCR-only extraction (no VL)")
    ocr_parser.add_argument("--urls", nargs="+", help="URLs to benchmark")
    ocr_parser.add_argument("--category", choices=["short", "long", "complex", "all"], help="Use preset URLs")
    ocr_parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    ocr_parser.add_argument("--runs", type=int, default=5, help="Measured runs per URL")
    ocr_parser.set_defaults(func=cmd_ocr_benchmark)

    # Compare OCR vs VL vs Hybrid
    compare_parser = subparsers.add_parser("compare", help="Compare OCR vs VL vs Hybrid")
    compare_parser.add_argument("--urls", nargs="+", help="URLs to benchmark")
    compare_parser.add_argument("--category", choices=["short", "long", "complex", "all"], help="Use preset URLs")
    compare_parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    compare_parser.add_argument("--runs", type=int, default=5, help="Measured runs per URL")
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if args.command is None:
        # Default: run screenshot benchmark with short URLs
        args.func = cmd_screenshot_benchmark
        args.urls = None
        args.category = "short"
        args.warmup = 2
        args.runs = 5

    args.func(args)


if __name__ == "__main__":
    main()
