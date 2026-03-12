from typing import Callable, Dict, Optional, Sequence, Union

import paddle

from .blocks import Dense, build_mlp

__all__ = ["Atomwise"]


class Atomwise(paddle.nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
        self,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Callable = paddle.nn.functional.silu,
        add_linear_nn: bool = False,
        output_scaling_factor: float = 1.0,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            add_linear_nn: whether to add a linear NN to the output of the MLP
        """
        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.add_linear_nn = add_linear_nn
        self.bias = bias
        self.output_scaling_factor = output_scaling_factor
        self.outnet = None

    def forward(
        self, desc: paddle.Tensor, batch: paddle.Tensor, training: bool = None
    ) -> paddle.Tensor:
        if self.n_in is None:
            self.n_in = desc.shape[1]
        else:
            assert self.n_in == desc.shape[1]
        if self.outnet is None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                bias=self.bias,
            )
            self.outnet = self.outnet.to(desc.device)
            if self.add_linear_nn:
                self.linear_nn = Dense(
                    self.n_in, self.n_out, bias=self.bias, activation=None
                )
                self.linear_nn = self.linear_nn.to(desc.device)
            else:
                self.linear_nn = None
        y = self.outnet(desc)
        if self.add_linear_nn:
            y += self.linear_nn(desc)
        return y * self.output_scaling_factor

    def __repr__(self):
        return f"Atomwise(n_in={self.n_in}, n_out={self.n_out}, n_hidden={self.n_hidden}, n_layers={self.n_layers}, bias={self.bias}, activation={self.activation.__name__})"
