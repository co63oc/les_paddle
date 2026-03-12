from typing import Callable, Optional, Sequence, Union

import numpy as np
import paddle

__all__ = ["build_mlp", "Dense"]
from typing import Callable, Union


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = paddle.nn.functional.silu,
    bias: bool = True,
) -> paddle.nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation, bias=bias)
        for i in range(n_layers - 1)
    ]
    layers.append(Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias))
    out_net = paddle.nn.Sequential(*layers)
    return out_net


class Dense(paddle.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, paddle.nn.Module] = paddle.nn.Identity(),
    ):
        """
        Fully connected linear layer with an optional activation function and batch normalization.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If False, the layer will not have a bias term.
            activation (Callable or nn.Module): Activation function. Defaults to Identity.
        """
        super().__init__()
        self.linear = paddle.compat.nn.Linear(in_features, out_features, bias)
        self.activation = activation
        if self.activation is None:
            self.activation = paddle.nn.Identity()

    def forward(self, input: paddle.Tensor):
        y = self.linear(input)
        y = self.activation(y)
        return y
