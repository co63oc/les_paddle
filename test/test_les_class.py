import sys

import paddle

sys.path.append("../")
import les
from les import Les

les = Les(les_arguments={})
paddle.manual_seed(0)
r = paddle.rand(10, 3) * 10
r.requires_grad_(requires_grad=True)
q = paddle.rand(10) * 2 - 1
box = paddle.tensor([10.0, 10.0, 10.0])
box_full = paddle.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
result = les(
    desc=r,
    positions=r,
    cell=box_full.unsqueeze(0),
    batch=None,
    compute_bec=True,
    bec_output_index=1,
)
print(result)
