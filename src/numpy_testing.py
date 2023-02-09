#!/usr/bin/env python3

import torch
import numpy as np
import triton
from torchviz import make_dot


def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(x.dtype)


# This has been tested as producing the same result as the pytorch GELU
def gelu(x):
    return (0.5 * x * (1.0 + np.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))))

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

class NumpyNN():

    def __init__(self, W: np.ndarray):
        self.W = W

    def forward(self, x: np.ndarray) -> np.ndarray:
        Z = np.matmul(x, self.W)
        z1, z2 = np.split(Z, 2, axis=1)
        # A = z1 * gelu(z2)
        # My guess is that dA = dz1 * 
        A = np.concatenate((z1, gelu(z2)), axis=1)
        return A

    def backward(self, x: np.ndarray, # [2, 2] this is the activation of the previous step, for the first step that's the input matrix
                 dA: np.ndarray # [2, 4] this is the gradient of this step, provided by the previous step of backprop (the previous layer)
                ) -> np.ndarray:
        """
        Backward pass of the linear layer with partial gelu activation.
        This returns the gradient of the linear layer's weights.
        """
        assert x.shape == (2, 2), f"Expected x to be [2, 2], got {x.shape}"
        dA1, dA2 = np.split(dA, 2, axis=1) # [2, 2] and [2, 2]
        Z = np.matmul(x, self.W) # [2, 4]
        _, z2 = np.split(Z, 2, axis=1) # [2, 2] and [2, 2]

        dz1 = dA1 # there's no activation function on this half
        dW1 = np.matmul(dz1.T, x).T
        # dW1 = np.split(np.matmul(dA.T, x).T, 2, axis=1)[0] # [2, 2]

        dz2 = dA2 * gelu_prime(z2) # [2, 2]
        dW2 = np.matmul(dz2.T, x).T # [2, 2]

        dW = np.concatenate((dW1, dW2), axis=1) # (2, 4)
        # assert dW.shape == self.W.shape, f"Expected dW to be {self.W.shape}, got {dW.shape}"
        return dW


class TorchNN():

    def __init__(self, d_model: int):
        self.linear = torch.nn.Linear(d_model, d_model * 2, bias=False)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.linear(x)
        z1, z2 = torch.chunk(Z, 2, dim=1)
        A = torch.cat((z1, self.gelu(z2)), 1)
        # A = z1 * self.gelu(z2)
        return A


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    d_model = 2
    batch_size = 2
    x = torch.randn(batch_size, d_model, device='cpu')  # input

    # PyTorch
    torch_nn = TorchNN(d_model)
    torch_forward = torch_nn.forward(x)
    # This is dA, or some upstream gradient that we must backpropagate
    grad = torch.ones(batch_size, d_model * 2, device='cpu')
    make_dot(torch_forward).render("test", format="png")
    torch_forward.backward(grad)
    torch_grad = torch_nn.linear.weight.grad
    assert torch_grad is not None
    torch_grad = torch_grad.T

    # Numpy
    W = torch_nn.linear.state_dict()["weight"].T
    assert torch_grad.shape == W.shape
    numpy_nn = NumpyNN(W.detach().numpy())
    numpy_forward = numpy_nn.forward(x.detach().numpy())
    numpy_grad = numpy_nn.backward(x.detach().numpy(), grad.detach().numpy())
    # assert numpy_grad.shape == W.shape

    triton.testing.assert_almost_equal(torch_forward.detach().numpy(), numpy_forward)
    triton.testing.assert_almost_equal(torch_grad, numpy_grad)
    print("Success!")


if __name__ == "__main__":
    main()
