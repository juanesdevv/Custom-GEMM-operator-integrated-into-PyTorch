import pytest
import torch

from src import gemm_blocked, gemm_naive


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


class TestGEMMCorrectness:
    def test_naive_float32(self):
        A = torch.randn(128, 64)
        B = torch.randn(64, 96)
        assert torch.allclose(gemm_naive(A, B), torch.matmul(A, B), atol=1e-5), (
            "Naive float32 mismatch"
        )

    def test_blocked_float32(self):
        A = torch.randn(128, 64)
        B = torch.randn(64, 96)
        assert torch.allclose(
            gemm_blocked(A, B, blockm=32, blockn=32, blockk=32),
            torch.matmul(A, B),
            atol=1e-5,
        ), "Blocked float32 mismatch"

    def test_naive_float64(self):
        A = torch.randn(64, 64, dtype=torch.float64)
        B = torch.randn(64, 64, dtype=torch.float64)
        assert torch.allclose(gemm_naive(A, B), torch.matmul(A, B), atol=1e-10)

    def test_blocked_float64(self):
        A = torch.randn(64, 64, dtype=torch.float64)
        B = torch.randn(64, 64, dtype=torch.float64)
        assert torch.allclose(gemm_blocked(A, B), torch.matmul(A, B), atol=1e-10)

    def test_edge_cases(self):
        A = torch.randn(8, 4).T.contiguous().T
        B = torch.randn(4, 8).T.contiguous().T
        assert torch.allclose(gemm_naive(A, B), torch.matmul(A, B), atol=1e-5)
        assert torch.allclose(gemm_blocked(A, B), torch.matmul(A, B), atol=1e-5)

        A = torch.randn(2, 5)
        B = torch.randn(5, 3)
        assert torch.allclose(gemm_naive(A, B), torch.matmul(A, B), atol=1e-5)
        assert torch.allclose(gemm_blocked(A, B), torch.matmul(A, B), atol=1e-5)

        A = torch.randn(70, 33)
        B = torch.randn(33, 50)
        assert torch.allclose(
            gemm_blocked(A, B, blockm=32, blockn=32, blockk=32),
            torch.matmul(A, B),
            atol=1e-5,
        )

    def test_mismatched_dims(self):
        A = torch.randn(3, 4)
        B = torch.randn(5, 2)
        with pytest.raises(RuntimeError):
            gemm_naive(A, B)
        with pytest.raises(RuntimeError):
            gemm_blocked(A, B)

    def test_single_element(self):
        A = torch.randn(1, 1)
        B = torch.randn(1, 1)
        assert torch.allclose(gemm_naive(A, B), torch.matmul(A, B), atol=1e-6)
        assert torch.allclose(gemm_blocked(A, B), torch.matmul(A, B), atol=1e-6)

    def test_larger_matrix(self):
        A = torch.randn(256, 256)
        B = torch.randn(256, 256)
        ref = torch.matmul(A, B)
        assert torch.allclose(gemm_naive(A, B), ref, atol=1e-4)
        assert torch.allclose(gemm_blocked(A, B), ref, atol=1e-4)


class TestAutograd:
    def test_gradcheck_naive(self):
        A = torch.randn(16, 32, dtype=torch.float64, requires_grad=True)
        B = torch.randn(32, 16, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(gemm_naive, (A, B), eps=1e-6, atol=1e-4)

    def test_gradcheck_blocked(self):
        A = torch.randn(16, 32, dtype=torch.float64, requires_grad=True)
        B = torch.randn(32, 16, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            gemm_blocked, (A, B, 64, 64, 64), eps=1e-6, atol=1e-4
        )

    def test_backward_values_naive(self):
        A = torch.randn(8, 16, requires_grad=True)
        B = torch.randn(16, 8, requires_grad=True)

        A_ref = A.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        loss_ref = torch.matmul(A_ref, B_ref).sum()
        loss_ref.backward()

        loss = gemm_naive(A, B).sum()
        loss.backward()

        assert torch.allclose(A.grad, A_ref.grad, atol=1e-5)
        assert torch.allclose(B.grad, B_ref.grad, atol=1e-5)


class TestFrameworkIntegration:
    def test_opcheck(self):
        A = torch.randn(64, 64)
        B = torch.randn(64, 64)
        from torch.library import opcheck

        opcheck(torch.ops.student_ops.gemm_naive, (A, B))
        opcheck(torch.ops.student_ops.gemm_blocked, (A, B, 32, 32, 32))

    def test_callable_from_torch_ops(self):
        A = torch.randn(32, 32)
        B = torch.randn(32, 32)
        out = torch.ops.student_ops.gemm_naive(A, B)
        assert out.shape == (32, 32)

    def test_output_dtype_preserved(self):
        for dt in [torch.float32, torch.float64]:
            A = torch.randn(16, 16, dtype=dt)
            B = torch.randn(16, 16, dtype=dt)
            assert gemm_naive(A, B).dtype == dt
            assert gemm_blocked(A, B).dtype == dt
