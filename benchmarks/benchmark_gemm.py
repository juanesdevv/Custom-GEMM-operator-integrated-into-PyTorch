import time

import numpy as np
import torch

from src.utils.cli import parse_args
from src.utils.helpers import load_cache, save_cache
from src.utils.plots import plot_all, print_table

CACHE_FILE = "benchmark_results.json"
PLOTS_DIR = "benchmark_plots"


def run_benchmark(func, A, B, *args, warmup: int = 3, runs: int = 10) -> float:
    for _ in range(warmup):
        _ = func(A, B, *args)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = func(A, B, *args)
        end = time.perf_counter()
        times.append(end - start)

    return float(np.median(times) * 1000.0)


def tflops(N: int, ms: float) -> float:
    flops = 2.0 * N**3
    return flops / (ms / 1000.0) / 1e12


def run_benchmarks(
    sizes: list[int],
    cache: dict,
    skip_naive: bool = False,
    force: bool = False,
    warmup: int = 3,
    runs: int = 10,
) -> dict:
    try:
        from src import gemm_blocked, gemm_naive
    except ImportError:
        print("ERROR: Could not import src. Build the extension first:")
        print("  pip install -e . --no-build-isolation")
        raise

    kernels = []
    if not skip_naive:
        kernels.append(("naive", lambda A, B: gemm_naive(A, B), []))
    kernels.append(("blocked", lambda A, B: gemm_blocked(A, B), []))
    kernels.append(("torch", lambda A, B: torch.matmul(A, B), []))

    for N in sizes:
        key = str(N)
        if key not in cache:
            cache[key] = {}

        torch.manual_seed(42)
        A = torch.randn(N, N)
        B = torch.randn(N, N)

        for name, fn, extra_args in kernels:
            if name in cache[key] and not force:
                print(
                    f"  N={N:5d}  {name:10s}  [cached] {cache[key][name]['ms']:.2f} ms"
                )
                continue

            print(f"  N={N:5d}  {name:10s}  running...", end="", flush=True)
            ms = run_benchmark(fn, A, B, *extra_args, warmup=warmup, runs=runs)
            tf = tflops(N, ms)
            cache[key][name] = {"ms": ms, "tflops": tf}
            print(f"  {ms:9.2f} ms  ({tf:.4f} TFLOP/s)")

        save_cache(CACHE_FILE, cache)

    return cache


def main():
    args = parse_args()

    import multiprocessing

    ncores = multiprocessing.cpu_count()
    print(f"CPU cores available: {ncores}")
    print(f"PyTorch version:     {torch.__version__}")
    print(f"OMP threads:         {torch.get_num_threads()}")
    print(f"Cache file:          {args.cache}")
    print(f"Plots directory:     {args.plots_dir}")
    print()

    cache = load_cache(args.cache)

    if not args.plots_only:
        print("Running benchmarks…")
        print(f"  Sizes   : {args.sizes}")
        print(f"  Warmup  : {args.warmup}  Runs: {args.runs}")
        print(f"  Skip naive: {args.skip_naive}")
        print()
        cache = run_benchmarks(
            sizes=args.sizes,
            cache=cache,
            skip_naive=args.skip_naive,
            force=args.force,
            warmup=args.warmup,
            runs=args.runs,
        )

    print_table(cache, args.sizes, args.skip_naive)
    plot_all(cache, args.sizes, args.plots_dir)
    print(f"\nDone. Plots saved to '{args.plots_dir}/'")


if __name__ == "__main__":
    main()
