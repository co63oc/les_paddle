import sys

import paddle

sys.path.append("../")
import les
from les.module import Ewald


def replicate_box(r, q, box, nx=2, ny=2, nz=2):
    """Replicate the simulation box nx, ny, nz times in each direction."""
    n_atoms = r.shape[0]
    replicated_r = []
    replicated_q = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = (
                    paddle.tensor([ix, iy, iz], dtype=r.dtype, device=r.device) * box
                )
                replicated_r.append(r + shift)
                replicated_q.append(q)
    replicated_r = paddle.cat(replicated_r)
    replicated_q = paddle.cat(replicated_q)
    new_box = paddle.tensor([nx, ny, nz], dtype=r.dtype, device=r.device) * box
    return replicated_r, replicated_q, new_box


ep = Ewald(dl=1.5, sigma=1)
paddle.manual_seed(0)
r = paddle.rand(100, 3) * 10
q = paddle.rand(100) * 2 - 1
box = paddle.tensor([10.0, 10.0, 10.0])
box_full = paddle.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
replicated_r, replicated_q, new_box = replicate_box(r, q, box, nx=2, ny=2, nz=2)
ew_1 = ep.compute_potential_triclinic(
    paddle.tensor(r), paddle.tensor(q).unsqueeze(1), paddle.tensor(box_full)
)
print(ew_1)
ew_2 = ep.compute_potential_triclinic(
    replicated_r, replicated_q.unsqueeze(1), paddle.tensor(box_full * 2.0)
)
print(ew_2 / 8.0)
