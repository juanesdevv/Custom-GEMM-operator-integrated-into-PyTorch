import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLORS = {
    "naive": "#e74c3c",  # red
    "blocked": "#2ecc71",  # green
    "torch": "#3498db",  # blue
}
LABELS = {
    "naive": "Naive (i-k-j + OMP)",
    "blocked": "Blocked (tiled + OMP)",
    "torch": "torch.matmul",
}


def _present_sizes(cache, sizes, key="ms"):
    return [N for N in sizes if cache.get(str(N), {})]


def plot_time_vs_n(cache: dict, sizes: list[int], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    present = _present_sizes(cache, sizes)

    for name, color in COLORS.items():
        ys = [cache.get(str(N), {}).get(name, {}).get("ms", None) for N in present]
        valid = [(x, y) for x, y in zip(present, ys) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(
                xs, ys, "o-", color=color, label=LABELS[name], linewidth=2, markersize=6
            )

    ax.set_xlabel("Matrix size N  (N×N square)", fontsize=12)
    ax.set_ylabel("Median time (ms)", fontsize=12)
    ax.set_title("Execution Time vs Matrix Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "time_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tflops_vs_n(cache: dict, sizes: list[int], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    present = _present_sizes(cache, sizes)

    for name, color in COLORS.items():
        ys = [cache.get(str(N), {}).get(name, {}).get("tflops", None) for N in present]
        valid = [(x, y) for x, y in zip(present, ys) if y is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(
                xs, ys, "o-", color=color, label=LABELS[name], linewidth=2, markersize=6
            )

    ax.set_xlabel("Matrix size N  (N×N square)", fontsize=12)
    ax.set_ylabel("TFLOP/s", fontsize=12)
    ax.set_title("Throughput (TFLOP/s) vs Matrix Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "tflops_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bar_chart(
    cache: dict, sizes: list[int], out_dir: str, target_n: int = 1024
) -> None:
    key = str(target_n)
    if key not in cache:
        print(f"  [bar chart] N={target_n} not in cache — skipping.")
        return

    names = [n for n in ["naive", "blocked", "torch"] if n in cache[key]]
    values = [cache[key][n]["tflops"] for n in names]
    colors = [COLORS[n] for n in names]
    labels = [LABELS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("TFLOP/s", fontsize=12)
    ax.set_title(
        f"Throughput Comparison at N={target_n}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"bar_tflops_N{target_n}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_speedup(cache: dict, sizes: list[int], out_dir: str) -> None:
    present = _present_sizes(cache, sizes)
    naive_present = [N for N in present if "naive" in cache.get(str(N), {})]
    if not naive_present:
        print("  [speedup chart] No naive results available — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(
        1.0,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=1.5,
        label=LABELS["naive"] + " (baseline = 1×)",
    )

    for name in ["blocked", "torch"]:
        ys = []
        xs = []
        for N in naive_present:
            ms_naive = cache[str(N)].get("naive", {}).get("ms")
            ms_this = cache[str(N)].get(name, {}).get("ms")
            if ms_naive and ms_this:
                ys.append(ms_naive / ms_this)
                xs.append(N)
        if xs:
            ax.plot(
                xs,
                ys,
                "o-",
                color=COLORS[name],
                label=LABELS[name],
                linewidth=2,
                markersize=7,
            )

    ax.set_xlabel("Matrix size N  (N×N square)", fontsize=12)
    ax.set_ylabel("Speedup over naive  (×)", fontsize=12)
    ax.set_title("Speedup Relative to Naive Kernel", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "speedup_vs_n.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_all(cache: dict, sizes: list[int], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print("\nGenerating plots...")
    plot_time_vs_n(cache, sizes, out_dir)
    plot_tflops_vs_n(cache, sizes, out_dir)
    plot_bar_chart(cache, sizes, out_dir, target_n=1024)
    avail = sorted([int(k) for k in cache if cache[k]])
    if avail and avail[-1] != 1024:
        plot_bar_chart(cache, sizes, out_dir, target_n=avail[-1])
    plot_speedup(cache, sizes, out_dir)


def print_table(cache: dict, sizes: list[int], skip_naive: bool) -> None:
    cols = []
    if not skip_naive:
        cols.append("naive")
    cols += ["blocked", "torch"]

    headers = ["N"] + [f"{c} ms" for c in cols] + [f"{c} TF/s" for c in cols]
    if not skip_naive and "naive" in cols and "blocked" in cols:
        headers.append("speedup blk/naive")

    print("\n" + "─" * 100)
    print(" | ".join(f"{h:>16}" for h in headers))
    print("─" * 100)

    for N in sizes:
        key = str(N)
        row = [f"{N:16d}"]
        for c in cols:
            ms = cache.get(key, {}).get(c, {}).get("ms", float("nan"))
            row.append(f"{ms:16.2f}")
        for c in cols:
            tf = cache.get(key, {}).get(c, {}).get("tflops", float("nan"))
            row.append(f"{tf:16.4f}")

        if not skip_naive and "naive" in cols and "blocked" in cols:
            ms_n = cache.get(key, {}).get("naive", {}).get("ms", float("nan"))
            ms_b = cache.get(key, {}).get("blocked", {}).get("ms", float("nan"))
            speedup = ms_n / ms_b if ms_b > 0 else float("nan")
            row.append(f"{speedup:16.2f}x")

        print(" | ".join(row))

    print("─" * 100 + "\n")
