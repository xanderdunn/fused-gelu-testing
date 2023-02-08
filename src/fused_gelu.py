#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_custom(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
        # by to get the element one row down (A has M rows)
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # L2 Cache Optimizations
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_left_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # offset the start by half of the number of columns
    b_right_ptrs = b_ptr + (N // 2) + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    x1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    x2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Matrix multiply A and B, accumulating the first half of columns into x1 and
    # the second half of columns into x2
    for k in range(0, K, BLOCK_SIZE_K): # type: ignore
        a = tl.load(a_ptrs)

        b_left = tl.load(b_left_ptrs)
        x1 += tl.dot(a, b_left)

        b_right = tl.load(b_right_ptrs)
        x2 += tl.dot(a, b_right)

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_left_ptrs += BLOCK_SIZE_K * stride_bk
        b_right_ptrs += BLOCK_SIZE_K * stride_bk

    # Element-wise multiply of x1 and gelu(x2)
    x2 = x1 * gelu_fast(x2)

    c = x2.to(tl.float16)

    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def gelu_fast(x):
    # From https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    # 0.5 * x * (1 + tanh(sqrt(2/pi)) * (x + 0.044715 * x**3))
    # 0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    return (
            0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
           )

@triton.jit
def gelu_fast_prime(x):
    # From https://arxiv.org/pdf/2104.02523.pdf
    term1 = 0.0356774 * x * x * x + 0.797885 * x
    denominator = tl.libdevice.cosh(term1)
    return (
            0.5 * tl.libdevice.tanh(term1) + 0.5 + (0.0535161 * x * x * x + 0.398942 * x) * (1.0 / (denominator * denominator))
           )


def matmul(a, b):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
            K % 16 == 0
            ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N // 2), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            )
    matmul_kernel_custom[grid]( # type: ignore
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # TODO: In production we would not want to hardcode these, but likely want to
        # find them via triton's autotuner.
        BLOCK_SIZE_M=32, # type: ignore
        BLOCK_SIZE_N=32, # type: ignore
        BLOCK_SIZE_K=32, # type: ignore
        GROUP_SIZE_M=8, # type: ignore
    )
    return c


class TorchNN():
    def __init__(self, d_model: int):
        self.linear = torch.nn.Linear(d_model, 8 * d_model, device='cuda', dtype=torch.float16)
        # TODO: Remove this once bias addition is added to the triton kernel
        with torch.no_grad():
            self.linear.bias.fill_(0.0)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        result = x1 * self.gelu(x2)
        return result


def main():
    print("Doing a forward pass: Linear layer -> split in two by column -> GELU on the second half -> element-wise multiply of the first and second halves")
    print("In both pytorch and triton to compare correctness...")
    batch_size = 512 
    d_model = 512 
    torch.manual_seed(0)
    # input to feed forward
    x = torch.randn((batch_size, d_model), device='cuda', dtype=torch.float16)

    torch_nn = TorchNN(d_model)
    torch_forward = torch_nn.forward(x)

    linear_weights = torch.transpose(torch_nn.linear.state_dict()["weight"], 0, 1).contiguous()
    # biases = torch_nn.linear.state_dict()["bias"]
    triton_forward = matmul(x, linear_weights)
    print(f"torch_forward={torch_forward}")
    print(f"triton_forward={triton_forward}")
    if triton.testing.allclose(triton_forward, torch_forward):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
    # TODO: Benchmark the two implementations across increasing sizes of matrices


if __name__ == "__main__":
    main()
