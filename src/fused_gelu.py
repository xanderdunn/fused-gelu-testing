#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
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
        x_ptr, W_ptr, z1_ptr, z2_ptr, dA_ptr, dW_ptr, dA1_ptr, dA2_ptr, dz1_ptr, dz2_ptr,
        M, N,
        stride_dA_half_m,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)

    # These operations can be done in blocks by row
    dz1 = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    dz2 = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)
    for _ in range(0, M, BLOCK_SIZE_M):
        dA = tl.load(dA_ptr + offsets)
        z1 = tl.load(z1_ptr + offsets)
        z2 = tl.load(z2_ptr + offsets)
        dA1 = dA * gelu_fast(z2)
        dz1 += dA1 # TODO: Is this accumulating as expected?

        dA2 = dA * z1
        dz2_piece = dA2 * gelu_fast_prime(z2)
        dz2 += dz2_piece
        tl.store(dA1_ptr + offsets, dA1)
        tl.store(dA2_ptr + offsets, dA2)
        tl.store(dz2_ptr + offsets, dz2_piece)
        offsets += BLOCK_SIZE_M * stride_dA_half_m

    dW1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dW2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # TODO: Do the matmul for dW1 and dW2, how to handle the different block sizes?

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

# From https://github.com/MarkTigchelaar/Tinman/blob/27a492c06105d550d7eacd2ca9fadc089d484c3a/src/neural_network_parts/activator.rs#L232
# In my testing this is a close approximation but not exact match of the backward pass of the pytorch GELU()
def gelu_prime(x: np.ndarray) -> np.ndarray:
    term1 = 0.0356774 * x * x * x;
    term2 = 0.398942 * x;
    term3 = 0.797885 * x;
    term4 = 0.0535161  * x * x * x;
    hyp_secant = 1 / np.cosh(term1 + term3);
    hyp_secant *= hyp_secant;
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

triton_z1 = None
triton_z2 = None
triton_dA1 = None
triton_dA2 = None
triton_dz2 = None
triton_dz1 = None

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
            # TODO: In production we would not want to hardcode these, but likely want to
            # find them via triton's autotuner.
            BLOCK_SIZE_M=32, # type: ignore
            BLOCK_SIZE_N=32, # type: ignore
            BLOCK_SIZE_K=16, # type: ignore
            GROUP_SIZE_M=8, # type: ignore
        )
        global triton_z1
        global triton_z2
        triton_z1 = z1
        triton_z2 = z2
        ctx.save_for_backward(x, W, z1, z2)
        return A

    def backward(ctx, dA: torch.Tensor) -> torch.Tensor:
        x, W, z1, z2 = ctx.saved_tensors
        M, K = x.shape
        K, N = W.shape
        dW = torch.empty(W.shape, device=x.device, dtype=x.dtype)
        dA1 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        dA2 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        dz1 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        dz2 = torch.zeros((K, N // 2), device=x.device, dtype=x.dtype)
        grid = lambda META: (
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
                )
        gelu_partial_layer_fused_backward[grid](
                x, W, z1, z2, dA, dW, dA1, dA2, dz1, dz2,
                M, N,
                stride_dA_half_m=dA1.stride(0),
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32, # type: ignore
        )
        global triton_dA1
        global triton_dA2
        global triton_dz1
        global triton_dz2
        triton_dA1 = dA1
        triton_dA2 = dA2
        triton_dz1 = dz1
        triton_dz2 = dz2
        return None, dW


class TorchNN():
    def __init__(self, d_model: int):
        # TODO: Add the bias back once the bias addition is added to the triton kernel
        self.linear = torch.nn.Linear(d_model, 8 * d_model, device='cuda', dtype=torch.float16, bias=False)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        result = x1 * self.gelu(x2)
        return result

partial_gelu = PartialGeluLayer.apply

def main():
    print("Doing a forward pass: Linear layer -> split in two by column -> GELU on the second half -> element-wise multiply of the first and second halves")
    print("In both pytorch and triton to compare correctness...")
    batch_size = 512
    d_model = 512
    torch.manual_seed(0)
    # input to feed forward
    x = torch.randn((batch_size, d_model), device='cuda', dtype=torch.float16)
    x.requires_grad_(True)

    torch_nn = TorchNN(d_model)
    torch_forward = torch_nn.forward(x)

    linear_weights = torch_nn.linear.state_dict()["weight"].T.contiguous()
    # biases = torch_nn.linear.state_dict()["bias"]
    triton_forward = partial_gelu(x, linear_weights)

    # Test that the portions of Z were stored correctly for retrieval by the backward pass
    Z = torch_nn.linear(x)
    torch_z1, torch_z2 = torch.chunk(Z, 2, dim=1)
    triton.testing.assert_almost_equal(torch_z1, triton_z1)
    triton.testing.assert_almost_equal(torch_z2, triton_z2)

    # print(f"torch_forward={torch_forward}")
    # print(f"triton_forward={triton_forward}")
    if triton.testing.allclose(triton_forward, torch_forward):
        print("Forward: ✅ Triton and Torch match")
    else:
        print("Forward: ❌ Triton and Torch differ")
    # TODO: Benchmark the two implementations across increasing sizes of matrices


    dA = torch.randn(batch_size, d_model * 8 // 2, device=x.device, dtype=x.dtype)
    torch_forward.backward(dA)
    torch_grad = torch_nn.linear.weight.grad
    assert torch_grad is not None
    torch_grad = torch_grad.T

    triton_grad = triton_forward.backward(dA)

    # Test that the individual components of the backward pass are correct
    torch_dA1 = dA * torch_nn.gelu(torch_z2)
    torch_dA2 = dA * torch_z1
    torch_dz1 = torch_dA1
    torch_dz2 = torch_dA2.detach().cpu().numpy() * gelu_prime(torch_z2.detach().cpu().numpy())
    # print(f"triton_dz1 = {triton_dz1}")
    triton.testing.assert_almost_equal(torch_dA1, triton_dA1)
    triton.testing.assert_almost_equal(torch_dA2, triton_dA2)
    triton.testing.assert_almost_equal(torch_dz2, triton_dz2)
    # triton.testing.assert_almost_equal(torch_dz1, triton_dz1)


if __name__ == "__main__":
    main()
