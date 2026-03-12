import sys

import paddle

sys.path.append("../")
import les
from les.module import BEC

paddle.manual_seed(0)
r = paddle.rand(10, 3) * 10
r.requires_grad_(requires_grad=True)
q = paddle.rand(10) * 2 - 1
box = paddle.tensor([10.0, 10.0, 10.0])
box_full = paddle.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
bec = BEC(remove_mean=False)
test = bec(q=q, r=r, cell=box_full.unsqueeze(0), batch=None, output_index=None)
print("BEC output shape:", test.shape)
print("BEC output:", test)
print("q", q)
