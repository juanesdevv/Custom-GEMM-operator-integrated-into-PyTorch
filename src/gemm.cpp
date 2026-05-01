#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>

template <typename scalar_t>
void gemm_naive_impl(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    int64_t M,
    int64_t K,
    int64_t N)
{
    const scalar_t* __restrict__ A_ptr = a.data_ptr<scalar_t>();
    const scalar_t* __restrict__ B_ptr = b.data_ptr<scalar_t>();
    scalar_t*       __restrict__ C_ptr = c.data_ptr<scalar_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            const scalar_t a_ik = A_ptr[i * K + k];
            const scalar_t* b_row = B_ptr + k * N;
            scalar_t*       c_row = C_ptr + i * N;
            for (int64_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

template <typename scalar_t>
void gemm_blocked_impl(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t blockm,
    int64_t blockn,
    int64_t blockk)
{
    const scalar_t* __restrict__ A_ptr = a.data_ptr<scalar_t>();
    const scalar_t* __restrict__ B_ptr = b.data_ptr<scalar_t>();
    scalar_t*       __restrict__ C_ptr = c.data_ptr<scalar_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < M; i0 += blockm) {
        int64_t i_end = std::min(i0 + blockm, M);

        for (int64_t k0 = 0; k0 < K; k0 += blockk) {
            int64_t k_end = std::min(k0 + blockk, K);

            for (int64_t j0 = 0; j0 < N; j0 += blockn) {
                int64_t j_end = std::min(j0 + blockn, N);

                for (int64_t i = i0; i < i_end; ++i) {
                    for (int64_t k = k0; k < k_end; ++k) {
                        const scalar_t a_ik = A_ptr[i * K + k];
                        const scalar_t* b_row = B_ptr + k * N + j0;
                        scalar_t*       c_row = C_ptr + i * N + j0;
                        int64_t jlen = j_end - j0;
                        for (int64_t j = 0; j < jlen; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor gemm_naive_cpu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cpu(), "Input 'a' must be on CPU.");
    TORCH_CHECK(b.is_cpu(), "Input 'b' must be on CPU.");
    TORCH_CHECK(a.dim() == 2, "Input 'a' must be a 2D tensor.");
    TORCH_CHECK(b.dim() == 2, "Input 'b' must be a 2D tensor.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "Input tensors must have the same dtype.");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Matrix dimensions do not match for multiplication.");

    a = a.contiguous();
    b = b.contiguous();

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    torch::Tensor c = torch::zeros({M, N}, a.options());

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "gemm_naive_cpu", [&] {
        gemm_naive_impl<scalar_t>(a, b, c, M, K, N);
    });
    return c;
}

torch::Tensor gemm_blocked_cpu(
    torch::Tensor a, torch::Tensor b,
    int64_t blockm = 64, int64_t blockn = 64, int64_t blockk = 64)
{
    TORCH_CHECK(a.is_cpu() && b.is_cpu(), "Inputs must be on CPU.");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D tensors.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                "Inputs must have the same dtype.");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match.");
    TORCH_CHECK(blockm > 0 && blockn > 0 && blockk > 0,
                "Block sizes must be positive.");

    a = a.contiguous();
    b = b.contiguous();

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    torch::Tensor c = torch::zeros({M, N}, a.options());

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "gemm_blocked_cpu", [&] {
        gemm_blocked_impl<scalar_t>(
            a, b, c, M, K, N, blockm, blockn, blockk);
    });
    return c;
}

TORCH_LIBRARY(student_ops, m) {
    m.def("gemm_naive(Tensor a, Tensor b) -> Tensor");
    m.def("gemm_blocked(Tensor a, Tensor b, int blockm=64, int blockn=64, int blockk=64) -> Tensor");
}

TORCH_LIBRARY_IMPL(student_ops, CPU, m) {
    m.impl("gemm_naive",   &gemm_naive_cpu);
    m.impl("gemm_blocked", &gemm_blocked_cpu);
}