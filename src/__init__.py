from pathlib import Path

import torch
from torch.autograd import Function

torch.ops.load_library(str(Path(__file__).resolve().parent.parent / "src.so"))


class GEMMNaiveFunction(Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return torch.ops.student_ops.gemm_naive(a, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_a = torch.ops.student_ops.gemm_naive(grad_output, b.T.contiguous())
        grad_b = torch.ops.student_ops.gemm_naive(a.T.contiguous(), grad_output)
        return grad_a, grad_b


def gemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GEMMNaiveFunction.apply(a, b)


@torch.library.register_fake("student_ops::gemm_naive")
def fake_gemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch._check(a.dim() == 2 and b.dim() == 2, lambda: "Inputs must be 2D")
    torch._check(a.size(1) == b.size(0), lambda: "Inner dimensions must match")
    return torch.empty(
        (a.size(0), b.size(1)),
        dtype=a.dtype,
        device=a.device,
        layout=a.layout,
        requires_grad=False,
    )


class GEMMBlockedFunction(Function):
    @staticmethod
    def forward(ctx, a, b, blockm, blockn, blockk):
        ctx.save_for_backward(a, b)
        ctx.block_params = (blockm, blockn, blockk)
        return torch.ops.student_ops.gemm_blocked(a, b, blockm, blockn, blockk)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        blockm, blockn, blockk = 64, 64, 64

        grad_a = torch.ops.student_ops.gemm_blocked(
            grad_output, b.T.contiguous(), blockm, blockk, blockn
        )
        grad_b = torch.ops.student_ops.gemm_blocked(
            a.T.contiguous(), grad_output, blockn, blockk, blockm
        )
        return grad_a, grad_b, None, None, None


def gemm_blocked(
    a: torch.Tensor,
    b: torch.Tensor,
    blockm: int = 64,
    blockn: int = 64,
    blockk: int = 64,
) -> torch.Tensor:
    return GEMMBlockedFunction.apply(a, b, blockm, blockn, blockk)


@torch.library.register_fake("student_ops::gemm_blocked")
def fake_gemm_blocked(
    a: torch.Tensor,
    b: torch.Tensor,
    blockm: int = 64,
    blockn: int = 64,
    blockk: int = 64,
) -> torch.Tensor:
    torch._check(a.dim() == 2 and b.dim() == 2, lambda: "Inputs must be 2D")
    torch._check(a.size(1) == b.size(0), lambda: "Inner dimensions must match")
    return torch.empty(
        (a.size(0), b.size(1)),
        dtype=a.dtype,
        device=a.device,
        layout=a.layout,
        requires_grad=False,
    )


__all__ = ["gemm_naive", "gemm_blocked"]
