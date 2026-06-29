#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200
"""Reference pipeline submission (Step 1) — validates the (H, tau) contract against the
REAL contest checker in `--mode test`. This is the correctness oracle, NOT the final
entry: it is PyTorch-heavy and exists only to confirm the CholeskyQR + modified-LU
reconstruction reproduces a passing geqrf-format factorization on the real generators.
The fused CUDA kernel replaces the hot stages next.
"""
import torch
from task import input_t, output_t

EPS32 = torch.finfo(torch.float32).eps


def _cholesky_qr(A, passes=2):
    M = A.mT @ A
    L, info = torch.linalg.cholesky_ex(M)
    R = L.mT
    ok = info == 0
    Rs = torch.where(ok.view(-1, 1, 1), R, torch.eye(A.shape[-1], device=A.device))
    Q = torch.linalg.solve_triangular(Rs, A, upper=True, left=False)
    if passes > 1:
        Q2, R2, ok2 = _cholesky_qr(Q, passes - 1)
        Q, R, ok = Q2, R2 @ R, ok & ok2
    return Q, R, ok


def _reconstruct_via_lu(Q, A):
    """Modified-LU reconstruction (Ballard-Demmel-Grigori). Q - S = L U; L = Householder
    vectors; tau = 2/||v||^2 makes householder_product exactly orthogonal; R = triu(Q~^T A)."""
    b, n, _ = Q.shape
    B = Q.clone()
    V = torch.zeros_like(Q)
    for i in range(n):
        diag = B[:, i, i]
        si = -torch.sign(diag)
        si = torch.where(si == 0, torch.ones_like(si), si)
        piv = diag - si
        B[:, i, i] = piv
        V[:, i, i] = 1.0
        if i + 1 < n:
            col = B[:, i + 1:, i] / piv.unsqueeze(-1)
            V[:, i + 1:, i] = col
            urow = B[:, i, i + 1:]
            B[:, i + 1:, i + 1:] -= col.unsqueeze(-1) * urow.unsqueeze(1)
    Vv = torch.tril(V, -1) + torch.eye(n, device=Q.device).expand_as(V)
    tau = 2.0 / (Vv * Vv).sum(dim=1)
    H_vecs = torch.tril(Vv, -1)
    Qt = torch.linalg.householder_product(H_vecs, tau)
    R = torch.triu(Qt.mT @ A)
    return H_vecs + R, tau


def custom_kernel(data: input_t) -> output_t:
    A = data
    n = A.shape[-1]
    Q, R, ok = _cholesky_qr(A, passes=2)
    I = torch.eye(n, device=A.device).expand_as(Q)
    defect = torch.linalg.matrix_norm((Q.mT @ Q - I).float(), ord=1, dim=(-2, -1))
    good = ok & (defect < 100 * n * EPS32)
    H, tau = _reconstruct_via_lu(Q, A)
    bad = ~good
    if bad.any():
        Hb, taub = torch.geqrf(A[bad])
        H = H.clone(); tau = tau.clone()
        H[bad], tau[bad] = Hb, taub
    return H, tau
