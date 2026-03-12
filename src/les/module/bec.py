from typing import Dict, Optional

import paddle

from ..util import grad

__all__ = ["BEC"]


class BEC(paddle.nn.Module):

    def __init__(self, remove_mean: bool = True, epsilon_factor: float = 1.0):
        super().__init__()
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor**0.5

    def forward(
        self,
        q: paddle.Tensor,
        r: paddle.Tensor,
        cell: paddle.Tensor,
        batch: Optional[paddle.Tensor] = None,
        output_index: Optional[int] = None,
    ) -> paddle.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)
        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"
        if batch is None:
            batch = paddle.zeros(n, dtype=paddle.int64, device=r.device)
        unique_batches = paddle.compat.unique(batch)
        all_P = []
        all_phases = []
        for i in unique_batches:
            mask = batch == i
            r_now, q_now = r[mask], q[mask]
            if self.remove_mean:
                q_now = q_now - paddle.mean(q_now, dim=0, keepdim=True)
            if cell is not None:
                box_now = cell[i]
            if cell is None or paddle.linalg.det(x=box_now) < 1e-06:
                polarization = paddle.sum(q_now * r_now, dim=0)
                phase = paddle.ones_like(r_now, dtype=paddle.complex64)
            else:
                polarization, phase = self.compute_pol_pbc(r_now, q_now, box_now)
            if output_index is not None:
                polarization = polarization[output_index]
                phase = phase[:, output_index]
            all_P.append(polarization * self.normalization_factor)
            all_phases.append(phase)
        P = paddle.stack(all_P, dim=0)
        phases = paddle.cat(all_phases, dim=0)
        bec_complex = grad(y=P, x=r)
        result = bec_complex * phases.unsqueeze(1).conj()
        return result.real()

    def compute_pol_pbc(self, r_now, q_now, box_now):
        r_frac = paddle.matmul(r_now, paddle.linalg.inv(x=box_now))
        phase = paddle.exp(1.0j * 2.0 * paddle.pi * r_frac)
        S = paddle.sum(q_now * phase, dim=0)
        polarization = paddle.matmul(box_now.to(S.dtype), S.unsqueeze(1)) / (
            1.0j * 2.0 * paddle.pi
        )
        return polarization.reshape(-1), phase

    def __repr__(self):
        return (
            f"BEC(remove_mean={self.remove_mean}, epsilon_factor={self.epsilon_factor})"
        )
