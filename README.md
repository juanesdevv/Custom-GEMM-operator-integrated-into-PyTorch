# Custom GEMM Operator Integrated into PyTorch

A high-performance CPU-based General Matrix Multiplication (GEMM) operator implemented in C++ with Python bindings, seamlessly integrated into PyTorch via custom extensions. This project demonstrates optimization techniques including loop tiling, cache locality improvements, and OpenMP parallelization.

## Features

- **Two GEMM Implementations:**
  - `gemm_naive`: Straightforward i-k-j loop order with OpenMP parallelization
  - `gemm_blocked`: Tiled/blocked algorithm optimized for CPU cache utilization with OpenMP

- **PyTorch Integration:** Full autograd support with proper gradient computation via custom `Function` classes

- **Comprehensive Testing:** Unit tests, gradient checks (gradcheck), and integration tests with PyTorch's opcheck

- **Benchmarking Suite:** Performance analysis tools with automatic plot generation (time vs size, TFLOP/s, speedup charts)

- **Multi-precision Support:** float32 and float64 tensors

## Requirements

- **Python:** ≥ 3.12
- **PyTorch:** ≥ 2.0.0
- **Build Tools:** GCC with C++17 support, OpenMP
- **Optional:** matplotlib, pandas for visualization and analysis

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/juanesdevv/Custom-GEMM-operator-integrated-into-PyTorch.git
cd gemm
```

### 2. Create and Activate Virtual Environment

Using `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Or using `uv` (recommended for speed):

```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch numpy pytest matplotlib pandas
```

Or install from `pyproject.toml`:

```bash
pip install -e .
```

### 4. Build the C++ Extension

The extension is built automatically via setuptools when installing in development mode:

```bash
pip install -e . --no-build-isolation
```

This will:

- Compile `src/gemm.cpp` with optimization flags (`-O3 -march=native -fopenmp`)
- Generate `src.so` (the compiled shared library)
- Register operators with PyTorch under the `student_ops` namespace

**Note:** The build requires a C++ compiler with OpenMP support (GCC 9+, Clang 10+, or MSVC 2019+).

## Usage

### Python API

```python
import torch
from src import gemm_naive, gemm_blocked

# Create random matrices
A = torch.randn(1024, 1024)
B = torch.randn(1024, 1024)

# Compute GEMM with naive kernel
C_naive = gemm_naive(A, B)

# Compute GEMM with blocked kernel (tiled algorithm)
C_blocked = gemm_blocked(A, B, blockm=64, blockn=64, blockk=64)

# Compare with PyTorch
C_torch = torch.matmul(A, B)

assert torch.allclose(C_naive, C_torch, atol=1e-5)
assert torch.allclose(C_blocked, C_torch, atol=1e-5)
```

### Gradient Computation

Both kernels support automatic differentiation:

```python
import torch
from src import gemm_naive, gemm_blocked

A = torch.randn(512, 512, requires_grad=True)
B = torch.randn(512, 512, requires_grad=True)

# Forward pass
C = gemm_blocked(A, B)

# Backward pass
loss = C.sum()
loss.backward()

print("Gradient of A shape:", A.grad.shape)
print("Gradient of B shape:", B.grad.shape)
```

### Running Tests

Execute the full test suite:

```bash
pytest tests/ -v
```

Run specific test classes:

```bash
pytest tests/test_student_gemm.py::TestGEMMCorrectness -v
pytest tests/test_student_gemm.py::TestAutograd -v
```

### Benchmarking

#### Run Full Benchmark

```bash
python -m benchmarks.benchmark_gemm
```

This benchmarks both kernels across matrix sizes: [64, 128, 256, 512, 1024, 2048, 4096, 8192]

#### Benchmark Options

```bash
# Run with custom matrix sizes
python -m benchmarks.benchmark_gemm --sizes 128 256 512 1024

# Force re-run (ignore cache)
python -m benchmarks.benchmark_gemm --force

# Skip slow naive kernel
python -m benchmarks.benchmark_gemm --skip-naive

# Regenerate plots from existing cache
python -m benchmarks.benchmark_gemm --plots-only

# Custom cache location and plots directory
python -m benchmarks.benchmark_gemm --cache my_results.json --plots-dir my_plots

# Show all available options
python -m benchmarks.benchmark_gemm --help
```

#### Output

The benchmark generates:

- **Console table:** Median times (ms) and throughput (TFLOP/s) for each size
- **Cached results:** `benchmark_results.json` — all measurements in JSON format
- **Plots:** Saved to `benchmark_plots/`
  - `time_vs_n.png` — Execution time vs matrix size
  - `tflops_vs_n.png` — Throughput (TFLOP/s) vs matrix size
  - `bar_tflops_N1024.png` — Throughput comparison at N=1024
  - `speedup_vs_n.png` — Speedup of blocked kernel relative to naive

**Example Output:**

```
CPU cores available: 8
PyTorch version:     2.0.0
OMP threads:         8

Running benchmarks…
  Sizes   : [256, 512, 1024]
  Warmup  : 3  Runs: 10
  Skip naive: False

────────────────────────────────────────────────────────────────────────────────────────────────────
              N |     naive ms |    blocked ms |      torch ms |     naive TF/s |    blocked TF/s |      torch TF/s | speedup blk/naive
────────────────────────────────────────────────────────────────────────────────────────────────────
            256 |         4.56 |          3.21 |          2.15 |         3.57 |         5.06 |         7.59 |             1.42x
            512 |        36.58 |         18.92 |         11.24 |         3.62 |         6.98 |        11.65 |             1.93x
           1024 |       292.45 |        142.18 |         89.31 |         3.64 |         7.47 |        12.10 |             2.06x
────────────────────────────────────────────────────────────────────────────────────────────────────

Generating plots…
  Saved: benchmark_plots/time_vs_n.png
  Saved: benchmark_plots/tflops_vs_n.png
  Saved: benchmark_plots/bar_tflops_N1024.png
  Saved: benchmark_plots/speedup_vs_n.png

Done. Plots saved to 'benchmark_plots/'
```

## Project Structure

```
gemm/
├── src/
│   ├── __init__.py              # PyTorch binding and autograd integration
│   ├── gemm.cpp                 # C++ implementations of gemm_naive and gemm_blocked
│   ├── src.so                   # Compiled shared library (auto-generated)
│   └── utils/
│       ├── cli.py               # Command-line argument parser
│       ├── helpers.py           # Cache I/O utilities
│       └── plots.py             # Matplotlib visualization functions
├── benchmarks/
│   ├── __init__.py
│   └── benchmark_gemm.py        # Main benchmark script
├── tests/
│   ├── __init__.py
│   └── test_student_gemm.py     # Comprehensive test suite
├── pyproject.toml               # Project metadata and build config
├── setup.py                     # Extension build configuration
├── README.md                    # This file
└── benchmark_results.json       # Cache of benchmark results (auto-generated)
```

## Implementation Details

### C++ Extension (src/gemm.cpp)

The extension exports two custom PyTorch operators:

#### `gemm_naive(A: Tensor, B: Tensor) → C: Tensor`

- Simple i-k-j loop with OpenMP parallelization
- Memory-efficient but low arithmetic intensity

#### `gemm_blocked(A: Tensor, B: Tensor, blockm: int, blockn: int, blockk: int) → C: Tensor`

- Tiled algorithm: breaks matrices into cache-sized blocks
- Improves L1/L2 cache hit rates
- Default block sizes: 64×64×64
- Significant speedup on modern CPUs

### Python Bindings (src/**init**.py)

- `GEMMNaiveFunction` / `GEMMBlockedFunction`: Custom autograd functions
- Gradients computed via GEMM transpose operations
- Fake implementations (FakeTensor) for AOT compilation support
- Clean Python API: `gemm_naive(A, B)` and `gemm_blocked(A, B, blockm, blockn, blockk)`

## Performance Tips

1. **Matrix Size:** Kernels perform best on larger matrices (N ≥ 256) due to parallelization overhead
2. **Thread Count:** Set `OMP_NUM_THREADS` to match physical cores (not hyperthreads for GEMM)
   ```bash
   OMP_NUM_THREADS=8 python -m benchmarks.benchmark_gemm
   ```
3. **Block Sizes:** Tune `blockm`, `blockn`, `blockk` based on L2 cache size:
   - Typical L2 cache: 256 KB per core
   - For float32: block size = sqrt(256 KB / 4 bytes / 3 matrices) ≈ 64
4. **Data Layout:** Ensure tensors are contiguous; use `.contiguous()` if needed

## Troubleshooting

### Build Failures

**Issue:** `ImportError: cannot import name 'BuildExtension'`

```bash
# Solution: Update PyTorch
pip install --upgrade torch
```

**Issue:** OpenMP not found during build

```bash
# Solution: Install libomp (macOS)
brew install libomp

# Or for Linux
sudo apt-get install libomp-dev
```

### Runtime Issues

**Issue:** `RuntimeError: Inner dimensions must match`

```python
# Solution: Check matrix dimensions
A = torch.randn(m, k)
B = torch.randn(k, n)  # B's first dim must equal A's second dim
C = gemm_blocked(A, B)  # ✓ Correct
```

**Issue:** Gradient computation differs from PyTorch

- Ensure float64 for gradcheck (float32 may have precision issues)
- Check that `atol`/`rtol` parameters in gradcheck are appropriate

## References

- [PyTorch Extension Documentation](https://pytorch.org/docs/stable/notes/extending.html)
- [Custom C++ Operators Guide](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
- [GEMM Optimization Techniques](https://www.cs.utexas.edu/~flame/BLASwiki/index.php/Main_Page)
- [OpenMP Parallel Programming](https://www.openmp.org/spec-html/5.0/openmpsu59.html)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please open issues or pull requests with improvements or bug fixes.

## Contact

For questions or feedback, please open an issue on GitHub.
