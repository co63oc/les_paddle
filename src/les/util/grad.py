from typing import Dict, List, Optional

import paddle


def grad(y: paddle.Tensor, x: paddle.Tensor, training: bool = True) -> paddle.Tensor:
    """
    a wrapper for the gradient calculation
    alow multiple dimensional and/or complex y
    y: [n_graphs, ] or [n_graphs, dim_y]
    x: [n_nodes, :]
    """
    if y.is_complex():
        get_imag = True
    else:
        get_imag = False
    if len(y.shape) == 1:
        grad_outputs: List[Optional[paddle.Tensor]] = [paddle.ones_like(y)]
        gradient_real = paddle.grad(
            outputs=[y],
            inputs=[x],
            grad_outputs=grad_outputs,
            retain_graph=training or get_imag,
            create_graph=training,
            allow_unused=True,
        )[0]
        assert gradient_real is not None, "Gradient real is None"
        if get_imag:
            gradient_imag = paddle.grad(
                outputs=[y / 1.0j],
                inputs=[x],
                grad_outputs=grad_outputs,
                retain_graph=training,
                create_graph=training,
                allow_unused=True,
            )[0]
            assert gradient_imag is not None, "Gradient imag is None"
        else:
            gradient_imag = paddle.tensor(0.0, dtype=x.dtype, device=x.device)
    else:
        dim_y = y.shape[1]
        grad_outputs: List[Optional[paddle.Tensor]] = [paddle.ones_like(y[:, 0])]
        grad_list_real = []
        for i in range(dim_y):
            g = paddle.grad(
                outputs=[y[:, i]],
                inputs=[x],
                grad_outputs=grad_outputs,
                retain_graph=training or i < dim_y - 1 or get_imag,
                create_graph=training,
                allow_unused=True,
            )[0]
            assert g is not None, f"Gradient real for channel {i} is None"
            grad_list_real.append(g)
        gradient_real = paddle.stack(grad_list_real, dim=2)
        if get_imag:
            grad_list_imag = []
            for i in range(dim_y):
                g = paddle.grad(
                    outputs=[y[:, i] / 1.0j],
                    inputs=[x],
                    grad_outputs=grad_outputs,
                    retain_graph=training or i < dim_y - 1,
                    create_graph=training,
                    allow_unused=True,
                )[0]
                assert g is not None, f"Gradient imag for channel {i} is None"
                grad_list_imag.append(g)
            gradient_imag = paddle.stack(grad_list_imag, dim=2)
        else:
            gradient_imag = paddle.tensor(0.0, dtype=x.dtype, device=x.device)
    if get_imag:
        return gradient_real + 1.0j * gradient_imag
    else:
        return gradient_real
