#!/usr/bin/env python3

import torch
import numpy as np
import triton


def gelu(x):
    return (
            0.5 * x * (1.0 + np.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
           )

def gelu_prime(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + 0.5 * x * (
        1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))**2 * np.sqrt(2 / np.pi) * (1 + 0.044715 * 3 * x**2)

# nn_forward(x):
    # return x * x

# nn_backward(x, grad):
    # return 2 * x


class NumpyNN():

    def __init__(self, W: np.ndarray):
        self.W = W

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.W)

    def backward(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the linear layer with partial gelu activation.
        This returns the gradient of the linear layer's weights.
        """
        return np.matmul(grad, x)


class TorchNN():

    def __init__(self, d_model: int):
        # self.W = W
        self.linear = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    d_model = 64
    batch_size = 64
    x = torch.randn(batch_size, d_model, device='cpu') # input
    # W = torch.randn(d_model, d_model, device='cpu', requires_grad=True) # weights

    # PyTorch
    torch_nn = TorchNN(d_model)
    torch_forward = torch_nn.forward(x)
    grad = torch.ones(batch_size, d_model, device='cpu')
    torch_forward.backward(grad)
    torch_grad = torch_nn.linear.weight.grad

    # Numpy
    W = torch_nn.linear.state_dict()["weight"].T
    # W = torch_nn.linear.state_dict()["weight"]
    numpy_nn = NumpyNN(W.detach().numpy())
    numpy_forward = numpy_nn.forward(x.detach().numpy())
    numpy_grad = numpy_nn.backward(x.detach().numpy(), grad.detach().numpy())

    triton.testing.assert_almost_equal(torch_forward.detach().numpy(), numpy_forward)
    triton.testing.assert_almost_equal(torch_grad, numpy_grad)
    print("Success!")


if __name__ == "__main__":
    main()
