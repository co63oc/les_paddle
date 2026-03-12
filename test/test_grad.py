import sys

import paddle

sys.path.append("../")
import les
from les.util import grad

paddle.manual_seed(0)
r = paddle.rand(8, 3) * 10
r.requires_grad_(requires_grad=True)
y1 = paddle.sum(r, dim=0)
y2 = paddle.sum(r**2.0, dim=0)
y = paddle.stack([y1, y2]).T
g = grad(x=r, y=y)
print("Gradient of y1 with respect to r:")
print(g)
