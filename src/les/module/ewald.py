from itertools import product
from typing import Dict, Optional

import numpy as np
import paddle

__all__ = ["Ewald"]


class Ewald(paddle.nn.Module):

    def __init__(
        self, dl=2.0, sigma=1.0, remove_self_interaction=True, norm_factor=90.0474
    ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma**2 / 2.0
        self.twopi = 2.0 * paddle.pi
        self.twopi_sq = self.twopi**2
        self.remove_self_interaction = remove_self_interaction
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(
        self,
        q: paddle.Tensor,
        r: paddle.Tensor,
        cell: paddle.Tensor,
        batch: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if q.dim() == 1:
            q = q.unsqueeze(1)
        n, d = r.shape
        assert d == 3, "r dimension error"
        assert n == q.size(0), "q dimension error"
        if batch is None:
            batch = paddle.zeros(n, dtype=paddle.int64, device=r.device)
        unique_batches = paddle.compat.unique(batch)
        results = []
        for i in unique_batches:
            mask = batch == i
            r_raw_now, q_now = r[mask], q[mask]
            if cell is not None:
                box_now = cell[i]
            if cell is None or paddle.linalg.det(x=box_now) < 1e-06:
                pot = self.compute_potential_realspace(r_raw_now, q_now)
            else:
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now)
            results.append(pot)
        return paddle.stack(results, dim=0).sum(dim=1)

    def compute_potential_realspace(self, r_raw, q):
        epsilon = 1e-06
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        paddle.diagonal(r_ij).add_(paddle.to_tensor(epsilon))
        r_ij_norm = paddle.norm(r_ij, dim=-1)
        convergence_func_ij = paddle.erf(x=r_ij_norm / self.sigma / 2.0**0.5)
        r_p_ij = 1.0 / r_ij_norm
        if q.dim() == 1:
            q = q.unsqueeze(1)
        n_node, n_q = q.shape
        pot = (
            q.unsqueeze(0)
            * q.unsqueeze(1)
            * r_p_ij.unsqueeze(2)
            * convergence_func_ij.unsqueeze(2)
        )
        mask = (
            ~paddle.eye(pot.shape[0], device=pot.device).to(paddle.bool).unsqueeze(-1)
        )
        mask = paddle.vstack(x=[mask.transpose(0, -1)] * pot.shape[-1]).transpose(0, -1)
        pot = pot[mask].sum().reshape([-1]) / self.twopi / 2.0
        if self.remove_self_interaction == False:
            pot += paddle.sum(q**2) / (self.sigma * self.twopi ** (3.0 / 2.0))
        return pot * self.norm_factor

    def compute_potential_triclinic(self, r_raw, q, cell_now):
        device = r_raw.device
        cell_inv = paddle.linalg.inv(x=cell_now)
        G = 2 * paddle.pi * cell_inv.T
        norms = paddle.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = paddle.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = paddle.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = paddle.arange(-Nk[2], Nk[2] + 1, device=device)
        nvec = (
            paddle.stack(paddle.meshgrid(n1, n2, n3, indexing="ij"), dim=-1)
            .reshape(-1, 3)
            .to(G.dtype)
        )
        kvec = nvec @ G
        k_sq = paddle.sum(kvec**2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask]
        k_sq = k_sq[mask]
        nvec = nvec[mask]
        non_zero = (nvec != 0).to(paddle.int32)
        first_non_zero = paddle.argmax(non_zero, dim=1)
        sign = paddle.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | (nvec == 0).all(dim=1)
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = paddle.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)
        k_dot_r = paddle.matmul(r_raw, kvec.T)
        if q.dim() == 1:
            q = q.unsqueeze(1)
        cos_k_dot_r = paddle.cos(k_dot_r)
        sin_k_dot_r = paddle.sin(k_dot_r)
        S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_sq = S_k_real**2 + S_k_imag**2
        kfac = paddle.exp(-self.sigma_sq_half * k_sq) / k_sq
        volume = paddle.linalg.det(x=cell_now)
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume
        if self.remove_self_interaction:
            pot -= paddle.sum(q**2) / (self.sigma * (2 * paddle.pi) ** 1.5)
        return pot * self.norm_factor

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"
