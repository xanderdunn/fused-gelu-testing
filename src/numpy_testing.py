#!/usr/bin/env python3

import torch
import numpy as np
import triton


def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(x.dtype)


# This has been tested as producing the same result as the pytorch GELU
def gelu(x):
    return (0.5 * x * (1.0 + np.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))))

# From https://github.com/MarkTigchelaar/Tinman/blob/27a492c06105d550d7eacd2ca9fadc089d484c3a/src/neural_network_parts/activator.rs#L232
# In my testing this is a close but not exact approximation of the backward pass of the pytorch GELU()
def gelu_prime(x: np.ndarray) -> np.ndarray:
    term1 = 0.0356774 * x * x * x;
    term2 = 0.398942 * x;
    term3 = 0.797885 * x;
    term4 = 0.0535161  * x * x * x;
    hyp_secant = 1 / np.cosh(term1 + term3);
    hyp_secant *= hyp_secant;
    return 0.5 * np.tanh(term1 + term3) + (term4 + term2) * hyp_secant + 0.5

class NumpyNN():

    def __init__(self, W: np.ndarray):
        self.W = W

    def forward(self, x: np.ndarray) -> np.ndarray:
        Z = np.matmul(x, self.W)
        A = gelu(Z)
        # A = relu(Z)
        return A

    def backward(self, x: np.ndarray, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass of the linear layer with partial gelu activation.
        This returns the gradient of the linear layer's weights.
        """
        Z = np.matmul(x, self.W)
        dZ = dA * gelu_prime(Z)
        # dZ = dA * relu_prime(Z)
        dW = np.dot(dZ.T, x)
        return dW


class TorchNN():

    def __init__(self, d_model: int):
        # self.W = W
        self.linear = torch.nn.Linear(d_model, d_model * 2, bias=False)
        self.gelu = torch.nn.GELU()
        # self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.linear(x)
        A = self.gelu(Z)
        # A = self.relu(Z)
        return A


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    d_model = 16
    batch_size = 16
    x = torch.randn(batch_size, d_model, device='cpu')  # input
    # W = torch.randn(d_model, d_model, device='cpu', requires_grad=True) # weights

    # PyTorch
    torch_nn = TorchNN(d_model)
    torch_forward = torch_nn.forward(x)
    grad = torch.randn(batch_size, d_model * 2, device='cpu')
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
