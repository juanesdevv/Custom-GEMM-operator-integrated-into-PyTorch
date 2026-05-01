import argparse

import matplotlib

matplotlib.use("Agg")

CACHE_FILE = "benchmark_results.json"
PLOTS_DIR = "benchmark_plots"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GEMM operators")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Matrix sizes to benchmark (square N×N)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs before timing",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed runs (median is reported)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-run all benchmarks",
    )
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="Skip the slow naive kernel (useful for quick profiling)",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Do not run benchmarks — regenerate plots from existing cache",
    )
    parser.add_argument(
        "--cache",
        default=CACHE_FILE,
        help=f"Path to JSON cache file (default: {CACHE_FILE})",
    )
    parser.add_argument(
        "--plots-dir",
        default=PLOTS_DIR,
        help=f"Directory for plot output (default: {PLOTS_DIR})",
    )
    return parser.parse_args()
