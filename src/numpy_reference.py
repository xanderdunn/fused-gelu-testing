#!/usr/bin/env python3

import torch
import numpy as np
import triton

"""
This is a reference implementation of the desired forward and backward passes in numpy
with correctness testing against the pytorch implementation.

Useful pieces:
https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
https://medium.com/analytics-vidhya/neural-network-mnist-classifier-from-scratch-using-numpy-library-94bbcfed7eae
https://cs231n.github.io/optimization-2/
https://stackoverflow.com/questions/57248777/backward-function-in-pytorch
https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc
https://goodboychan.github.io/python/datacamp/deep_learning/2020/07/21/02-Optimizing-a-neural-network-with-backward-propagation.html
"""



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
        return z1 * gelu(z2)

    # Key insight here is that the backward pass on an element-wise matix multiplication is 
    # the element-wise multiplication with the other matrix
    # A = z1 * gelu(z2)
    # dA1 = dA * gelu(z2)
    # dA2 = dA * z1
    def backward(self, x: np.ndarray, # this is the activation of the previous step, for the first step that's the input matrix
                 dA: np.ndarray # this is the gradient of this step, provided by the previous step of backprop (the previous layer)
                ) -> np.ndarray:
        """
        Backward pass of the linear layer with partial gelu activation.
        This returns the gradient of the linear layer's weights.
        """
        # This is a duplicated matmul and would instead ideally be gotten from storage elsewhere
        Z = np.matmul(x, self.W)
        z1, z2 = np.split(Z, 2, axis=1)

        dA1 = dA * gelu(z2)
        dz1 = dA1 # there's no activation function on this half
        dW1 = np.matmul(dz1.T, x).T

        dA2 = dA * z1
        dz2 = dA2 * gelu_prime(z2)
        dW2 = np.matmul(dz2.T, x).T

        dW = np.concatenate((dW1, dW2), axis=1)
        return dW


class TorchNN():

    def __init__(self, d_model: int):
        self.linear = torch.nn.Linear(d_model, d_model * 8, bias=False)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.linear(x)
        assert not torch.isnan(Z).any()
        assert not torch.isinf(Z).any()
        z1, z2 = torch.chunk(Z, 2, dim=1)
        A = z1 * self.gelu(z2)
        return A


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    sizes = [2**i for i in range(6, 12)]
    print(f"sizes = {sizes}")
    for size in sizes:
        d_model = size
        batch_size = size
        x = torch.randn(batch_size, d_model, device='cpu')  # input

        # PyTorch
        torch_nn = TorchNN(d_model)
        torch_forward = torch_nn.forward(x)
        # This is dA, or some upstream gradient that we must backpropagate
        grad = torch.randn(batch_size, d_model * 8 // 2, device='cpu')
        torch_forward.backward(grad)
        torch_grad = torch_nn.linear.weight.grad
        assert torch_grad is not None
        assert not torch.isnan(torch_grad).any()
        assert not torch.isinf(torch_grad).any()
        torch_grad = torch_grad.T

        # Numpy
        W = torch_nn.linear.state_dict()["weight"].T
        assert torch_grad.shape == W.shape
        numpy_nn = NumpyNN(W.detach().numpy())
        numpy_forward = numpy_nn.forward(x.detach().numpy())
        numpy_grad = numpy_nn.backward(x.detach().numpy(), grad.detach().numpy())
        assert numpy_grad.shape == W.shape
        assert not np.isnan(numpy_grad).any()
        assert not np.isinf(numpy_grad).any()

        triton.testing.assert_almost_equal(torch_forward.detach().numpy(), numpy_forward)
        triton.testing.assert_almost_equal(torch_grad, numpy_grad, decimal=1)
        print(f"({size}, {size}) Success!")


if __name__ == "__main__":
    main()
