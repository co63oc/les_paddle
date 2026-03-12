import sys

sys.path.append("/nfs/github/co63oc/les_paddle")
import paddle
from paddle_utils import *

"""basic scatter_sum operations from torch_scatter from
https://github.com/mir-group/pytorch_runstats/blob/main/torch_runstats/scatter_sum.py
Using code from https://github.com/rusty1s/pytorch_scatter, but cut down to avoid a dependency.
PyTorch plans to move these features into the main repo, but until then,
to make installation simpler, we need this pure python set of wrappers
that don't require installing PyTorch C++ extensions.
See https://github.com/pytorch/pytorch/issues/63780.
"""
from typing import Optional


def _broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@paddle.jit.to_static
def scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.size == 0:
            size[dim] = 0
        else:
            size[dim] = int(index._max()) + 1
        out = paddle.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
