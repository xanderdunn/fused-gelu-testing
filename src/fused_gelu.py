#!/usr/bin/env python3

from typing import Tuple

import torch
import triton
import triton.language as tl
import numpy as np

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
# FIXME: This can produce a CUDA illegal access out of bounds error if d_model and batch_size aren't a power of 2
def gelu_partial_layer_fused_forward(
        # Pointers to matrices
        x_ptr, W_ptr, A_ptr, z1_ptr, z2_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. stride_xm is how much to increase x_ptr
        # by to get the element one row down (A has M rows)
        stride_xm, stride_xk,
        stride_Wk, stride_Wn,
        stride_Am, stride_An,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ):
    """
    1) Compute Z = matmul(x, W) into two halves, Z1 and Z2
    2) Compute element-wise multiply A = Z1 * gelu(Z2), this is the activation
    3) Return A
    x has shape (M, K), W has shape (K, N) and A has shape (M, N//2)
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

    # Create pointers for the first blocks of x and W.
    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_Wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    W_left_ptrs = W_ptr + (offs_k[:, None] * stride_Wk + offs_Wn[None, :] * stride_Wn)
    # offset the start by half of the number of columns
    W_right_ptrs = W_ptr + (N // 2) + (offs_k[:, None] * stride_Wk + offs_Wn[None, :] * stride_Wn)

    z1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    z2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Matrix multiply A and B, accumulating the first half of columns into x1 and
    # the second half of columns into x2
    for _ in range(0, K, BLOCK_SIZE_K): # type: ignore
        x = tl.load(x_ptrs)

        W_left = tl.load(W_left_ptrs)
        z1 += tl.dot(x, W_left)

        W_right = tl.load(W_right_ptrs)
        z2 += tl.dot(x, W_right)

        # Advance the ptrs to the next K block
        x_ptrs += BLOCK_SIZE_K * stride_xk
        W_left_ptrs += BLOCK_SIZE_K * stride_Wk
        W_right_ptrs += BLOCK_SIZE_K * stride_Wk

    # Store z1 in z1_ptr
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets = stride_Am * offs_cm[:, None] + stride_An * offs_cn[None, :]
    outs_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N // 2)
    tl.store(z1_ptr + offsets, z1, mask=outs_mask)
    # Store z2 in z2_ptr
    tl.store(z2_ptr + offsets, z2, mask=outs_mask)

    # Element-wise multiply of z1 and gelu(z2)
    z2 = gelu_fast(z2)
    c = z1 * z2
    c = c.to(tl.float16)

    # Write back the block of the output matrix C
    tl.store(A_ptr + offsets, c, mask=outs_mask)

@triton.jit
def gelu_partial_layer_fused_backward(
        z1_ptr, z2_ptr, dA_ptr, dz1_ptr, dz2_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    dA = tl.load(dA_ptr + offsets)
    z1 = tl.load(z1_ptr + offsets)
    z2 = tl.load(z2_ptr + offsets)
    dA1 = dA * gelu_fast(z2)
    dz1 = dA1

    dA2 = dA * z1
    dz2 = dA2 * gelu_fast_prime(z2)
    # tl.store(dA1_ptr + offsets, dA1)
    # tl.store(dA2_ptr + offsets, dA2)
    tl.store(dz2_ptr + offsets, dz2)
    tl.store(dz1_ptr + offsets, dz1)

@triton.jit
# TODO: Combine this with the gelu_partial_layer_fused_backward kernel
def gelu_partial_layer_fused_backward_matmul(
        dz1_ptr, b_ptr, dz2_ptr, dW_ptr,
        P, R, Q,
        stride_dzm, stride_dzk,
        stride_xk, stride_xn,
        stride_Wm, stride_Wn,
        BLOCK_SIZE_P: tl.constexpr,
        BLOCK_SIZE_R: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        GROUP_SIZE_P: tl.constexpr,
        ):
    # L2 Cache Optimizations
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(P, BLOCK_SIZE_P)
    num_pid_n = tl.cdiv(R, BLOCK_SIZE_R)
    num_pid_in_group = GROUP_SIZE_P * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_P
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_P)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks
    offs_dzm = pid_m * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_xn = pid_n * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_k = tl.arange(0, BLOCK_SIZE_Q)
    # Regular: ptrs = base + i[:, None]*stride_i + j[None, :]
    # Transposed: ptrs = base + i[:, None] + j[None, :]*stride_j
    # Transposed: ptrs = base + i[:, None]*stride_i + j[None, :]*stride_j, Compiler will figure it out if one of stride_i or stride_j is 1
    dz1_ptrs = dz1_ptr + (offs_dzm[:, None] * stride_dzm + offs_k[None, :] * stride_dzk)
    dz2_ptrs = dz2_ptr + (offs_dzm[:, None] * stride_dzm + offs_k[None, :] * stride_dzk)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_xk + offs_xn[None, :] * stride_xn)

    # Iterate to compute a block of the dW1 and dW2 matrices
    accumulator_dW1 = tl.zeros((BLOCK_SIZE_P, BLOCK_SIZE_R), dtype=tl.float32)
    accumulator_dW2 = tl.zeros((BLOCK_SIZE_P, BLOCK_SIZE_R), dtype=tl.float32)
    for _ in range(0, Q, BLOCK_SIZE_Q):
        dz1 = tl.load(dz1_ptrs)
        b = tl.load(b_ptrs)
        dz2 = tl.load(dz2_ptrs)
        accumulator_dW1 += tl.dot(dz1, b)
        accumulator_dW2 += tl.dot(dz2, b)
        dz1_ptrs += BLOCK_SIZE_Q * stride_dzk
        dz2_ptrs += BLOCK_SIZE_Q * stride_dzk
        b_ptrs += BLOCK_SIZE_Q * stride_xk
    dW1 = accumulator_dW1.to(tl.float16)
    dW2 = accumulator_dW2.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix dW1 and dW2
    # Store the transpose of dW1 and dW2 into the same dW output tensor
    offs_Wm = pid_m * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_Wn = pid_n * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    # To transpose storage, simply swap the strides multiplied by the rows and columns
    # Useful example here: https://github.com/Jokeren/triton-samples/blob/main/transpose-2d.py
    # Must offset dW2 by P becuase it's the second half of every row
    dW_ptrs = dW_ptr + P + stride_Wn * offs_Wm[:, None] + stride_Wm * 2 * offs_Wn[None, :]
    out_mask = (offs_Wm[:, None] < P) & (offs_Wn[None, :] < R)
    # tl.store(dW2_ptrs, dW2, mask=out_mask)
    tl.store(dW_ptrs, dW2, mask=out_mask)

    # Transpose
    # dW1_ptrs = dW1_ptr + stride_Wn * offs_Wm[:, None] + stride_Wm * offs_Wn[None, :]
    dW_ptrs = dW_ptr + stride_Wn * offs_Wm[:, None] + stride_Wm * 2 * offs_Wn[None, :]
    # tl.store(dW1_ptrs, dW1, mask=out_mask)
    tl.store(dW_ptrs, dW1, mask=out_mask)

@triton.jit
def gelu_fast(x):
    """
    From https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    0.5 * x * (1 + tanh(sqrt(2/pi)) * (x + 0.044715 * x**3))
    0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    """
    return (
            0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
           )

# This is used solely for debugging and correctness testing
def gelu_prime(x: np.ndarray) -> np.ndarray:
    """
    From https://github.com/MarkTigchelaar/Tinman/blob/27a492c06105d550d7eacd2ca9fadc089d484c3a/src/neural_network_parts/activator.rs#L232
    In my testing this is a close approximation but not exact match of the backward pass of the pytorch GELU()
    """
    term1 = 0.0356774 * x * x * x
    term2 = 0.398942 * x
    term3 = 0.797885 * x
    term4 = 0.0535161  * x * x * x
    hyp_secant = 1 / np.cosh(term1 + term3)
    hyp_secant *= hyp_secant
    return 0.5 * np.tanh(term1 + term3) + (term4 + term2) * hyp_secant + 0.5

@triton.jit
def gelu_fast_prime(x):
    """
    From https://github.com/MarkTigchelaar/Tinman/blob/27a492c06105d550d7eacd2ca9fadc089d484c3a/src/neural_network_parts/activator.rs#L232
    In my testing this is a close approximation but not exact match of the backward pass of the pytorch GELU()
    """
    term1 = 0.0356774 * x * x * x
    term2 = 0.398942 * x
    term3 = 0.797885 * x
    term4 = 0.0535161  * x * x * x
    hyp_secant = 1 / tl.libdevice.cosh(term1 + term3)
    hyp_secant *= hyp_secant
    return 0.5 * tl.libdevice.tanh(term1 + term3) + (term4 + term2) * hyp_secant + 0.5

# TODO: Remove these for production. 
# These are for debugging only to check the values
# against the pytorch reference implementation
# triton_z1 = None
# triton_z2 = None
# triton_dA1 = None
# triton_dA2 = None
# triton_dz2 = None
# triton_dz1 = None
# triton_dW1 = None
# triton_dW2 = None
triton_dW = None

class PartialGeluLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # checks constraints
        assert x.shape[1] == W.shape[0], "incompatible dimensions"
        assert x.is_contiguous(), "matrix A must be contiguous"
        assert W.is_contiguous(), "matrix B must be contiguous"
        M, K = x.shape
        K, N = W.shape
        assert (
                K % 16 == 0
                ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
        # allocates output
        A = torch.empty((M, N // 2), device=x.device, dtype=x.dtype)
        z1 = torch.empty((M, N // 2), device=x.device, dtype=x.dtype) # same dimensions as the weights
        z2 = torch.empty((M, N // 2), device=x.device, dtype=x.dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
                )
        gelu_partial_layer_fused_forward[grid]( # type: ignore
            x, W, A, z1, z2,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            A.stride(0), A.stride(1),
            # In production we would not want to hardcode these, but likely want to
            # find them via triton's autotuner.
            # BLOCK_SIZE_M=32, # type: ignore
            # BLOCK_SIZE_N=64, # type: ignore
            # BLOCK_SIZE_K=16, # type: ignore
            # GROUP_SIZE_M=8, # type: ignore
        )
        # global triton_z1
        # global triton_z2
        # triton_z1 = z1
        # triton_z2 = z2
        ctx.save_for_backward(x, W, z1, z2)
        return A

    @staticmethod
    def backward(ctx, dA: torch.Tensor) -> Tuple[None, torch.Tensor]:
        x, W, z1, z2 = ctx.saved_tensors
        _, K = x.shape
        K, N = W.shape
        dW = torch.empty(W.shape, device=x.device, dtype=x.dtype)
        # dA1 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        # dA2 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        dz1 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        dz2 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        assert(dz1.T.shape[1] == x.shape[0])
        assert(dz1.T.shape == dz2.T.shape)
        P, Q = dz1.T.shape
        Q, R = x.shape
        dW = torch.zeros(R, P * 2, device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(dz1.numel(), meta['BLOCK_SIZE']),)
        gelu_partial_layer_fused_backward[grid]( # type: ignore
                z1, z2, dA, dz1, dz2,
                BLOCK_SIZE=512
        )
        grid = lambda META: (
                triton.cdiv(P, META['BLOCK_SIZE_P']) * triton.cdiv(R, META['BLOCK_SIZE_R']),
                )
        gelu_partial_layer_fused_backward_matmul[grid]( # type: ignore
            dz1, x, dz2, dW,
            P, R, Q,
            dz1.T.stride(0), dz1.T.stride(1),
            x.stride(0), x.stride(1),
            dW.stride(0) // 2, dW.stride(1),
            BLOCK_SIZE_P=32, # type: ignore
            BLOCK_SIZE_R=32, # type: ignore
            BLOCK_SIZE_Q=16, # type: ignore
            GROUP_SIZE_P=8, # type: ignore
        )
        # TODO: Remove for production. This is for debugging only:
        # Store the outputs to compare to pytorch
        # global triton_dA1
        # global triton_dA2
        # global triton_dz1
        # global triton_dz2
        global triton_dW
        # triton_dA1 = dA1
        # triton_dA2 = dA2
        # triton_dz1 = dz1
        # triton_dz2 = dz2
        triton_dW = dW
        return None, dW


class TorchNN():
    def __init__(self, d_model: int):
        # TODO: Add the bias back once the bias addition is added to the triton kernel
        self.linear = torch.nn.Linear(d_model, 8 * d_model, device='cuda', dtype=torch.float16, bias=False)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.linear(x)
        z1, z2 = torch.chunk(Z, 2, dim=(Z.ndim - 1))
        A = z1 * self.gelu(z2)
        return A

partial_gelu = PartialGeluLayer.apply

def check_correctness():
    print("Linear layer -> split in two by column -> GELU on the second half -> element-wise multiply of the first and second halves")
    print("In both pytorch and triton to compare correctness...")
    sizes = [2**i for i in range(6, 12)]
    print("sizes = ", sizes)
    for size in sizes:
        batch_size = size # TODO: The backprop only works if this is square, where batch_size == d_model
        d_model = size
        torch.manual_seed(0)
        # input to feed forward
        x = torch.randn((batch_size, d_model), device='cuda', dtype=torch.float16)
        x.requires_grad_(True)

        # ptyroch
        torch_nn = TorchNN(d_model)
        torch_forward = torch_nn.forward(x)

        # TODO: Remove these debug checks for production
        # assert torch_nn.z1 is not None
        # assert torch_nn.z2 is not None
        # assert not torch.isnan(torch_nn.z1).any()
        # assert not torch.isinf(torch_nn.z1).any()
        # This is some arbitrary gradient passed from the previous layer
        dA = torch.randn(batch_size, d_model * 8 // 2, device=x.device, dtype=x.dtype)
        torch_forward.backward(dA)

        linear_weights = torch_nn.linear.state_dict()["weight"].T.contiguous()
        triton_forward = partial_gelu(x, linear_weights)

        # Test that the portions of Z were stored correctly for retrieval by the backward pass
        # triton.testing.assert_almost_equal(torch_nn.z1, triton_z1)
        # triton.testing.assert_almost_equal(torch_nn.z2, triton_z2)

        triton.testing.assert_almost_equal(triton_forward, torch_forward)
        print(f"Forward ({size}, {size}): ??? Triton and Torch match")

        torch_grad = torch_nn.linear.weight.grad
        assert torch_grad is not None
        torch_grad = torch_grad.T

        triton_forward.backward(dA)

        # Test that the individual components of the backward pass are correct
        # torch_dA1 = dA * torch_nn.gelu(torch_nn.z2)
        # torch_dA2 = dA * torch_nn.z1
        # torch_dz1 = torch_dA1
        # torch_dz2 = torch_dA2.detach().cpu().numpy() * gelu_prime(torch_nn.z2.detach().cpu().numpy())

        # triton.testing.assert_almost_equal(torch_dA1, triton_dA1)
        # triton.testing.assert_almost_equal(torch_dA2, triton_dA2)
        # triton.testing.assert_almost_equal(torch_dz1, triton_dz1)
        # triton.testing.assert_almost_equal(torch_dz2, triton_dz2)

        # torch_dW1_computed = torch.matmul(torch_dz1.T, x).T
        torch_dW1, torch_dW2 = torch.chunk(torch_grad, 2, dim=1)
        # triton.testing.assert_almost_equal(torch_dW1, torch_dW1_computed, decimal=1)

        # triton_dW1_computed = torch.matmul(triton_dz1.T, x).T
        # precision lossiness reduces the degree to which our results are identical
        # so reduce the precision of the comparison to 1 decimal
        # triton.testing.assert_almost_equal(triton_dW1_computed, torch_dW1, decimal=1)

        assert triton_dW is not None

        triton_dW1, triton_dW2 = torch.chunk(triton_dW, 2, dim=1)
        triton.testing.assert_almost_equal(torch_dW1, triton_dW1, decimal=1)
        triton.testing.assert_almost_equal(torch_dW2, triton_dW2, decimal=1)
        triton_dW_computed = torch.concat((triton_dW1, triton_dW2), dim=1)
        triton.testing.assert_almost_equal(torch_grad, triton_dW_computed, decimal=1)
        triton.testing.assert_almost_equal(torch_grad, triton_dW, decimal=1)
        print(f"Backward ({size}, {size}): ??? Triton and Torch match\n")
        # TODO: Benchmark the two implementations across increasing sizes of matrices

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an x-axis for the plot
        x_vals = [2**i for i in range(5, 13)],
        # x_vals=[
            # 128 * i for i in range(2, 33)
        # ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['pytorch', 'triton'],
        # label name for the lines
        line_names=["PyTorch", "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="partial-gelu-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"mode": "backward"},
    )
)


def benchmark(size, provider: str, mode: str):
    print(f"{provider} benchmarking {mode} pass on ({size}, {size})...")
    torch_nn = TorchNN(size)
    linear_weights = torch_nn.linear.state_dict()["weight"].T.contiguous()
    x = torch.randn((size, size), device='cuda', dtype=torch.float16)
    if provider == 'pytorch':
        forward_f = lambda: torch_nn.forward(x)
    elif provider == 'triton':
        forward_f = lambda: partial_gelu(x, linear_weights)
    else:
        assert(False), "Got invalid provider {}".format(provider)
    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(forward_f)
    elif mode == "backward":
        x.requires_grad_(True)
        forward = forward_f()
        dA = torch.randn(size, size * 8 // 2, device=x.device, dtype=x.dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: forward.backward(dA, retain_graph=True))
    else:
        assert(False), "Got invalid mode {}".format(mode)
    # TODO: This perf function should be better estimated for the forward and backward passes
    def perf(ms):
        return 2 * size * size * size  * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def run_benchmarks():
    print("Running benchmark...")
    benchmark.run(print_data=True, save_path="./")


def main():
    check_correctness()
    run_benchmarks()


if __name__ == "__main__":
    main()
