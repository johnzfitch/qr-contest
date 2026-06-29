#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200
"""
probe_a1_fp32.py — a1 with FP32-INTERNAL panel (candidate-a panel attack).

Purpose: get the OFFICIAL qr_v2 geomean for the pure blocked-Householder baseline
(no Ozaki, no low-bit trailing — fp64 internal panel via torch.geqrf, plain-GEMM
trailing). Layer-4 measured this at ~103.760 ms geomean on our perf harness; this
confirms where it actually lands on the board so the panel-attack design (the
panel is ~90% of runtime) has a real baseline to beat.

Faithful to qr_hh_portfolio.py ARCH 1 = ("fp32", 1, 1, 64, False):
  work = A.float()  # fp32 panel (candidate-a: drop fp64); per-panel torch.geqrf; trailing <- trailing - V T^T (V^T trailing)
  as plain double GEMMs; nb=64; no edge guard; n<=64 -> straight geqrf.
"""
import torch
from task import input_t, output_t

NB = 64
SMALL_N = 64


def _build_T(V, tau):
    G = V.transpose(-2, -1) @ V
    kb = V.shape[-1]
    M = torch.triu(G, 1) + torch.diag_embed(1.0 / tau)
    eye = torch.eye(kb, dtype=V.dtype, device=V.device).expand(V.shape[0], kb, kb)
    return torch.linalg.solve_triangular(M, eye, upper=True)


def blocked_hh(A):
    B, m, n = A.shape
    work = A.float()  # fp32 panel (candidate-a: drop fp64)
    H = work.clone()
    tau = torch.zeros(B, n, dtype=work.dtype, device=work.device)
    for k in range(0, n, NB):
        kb = min(NB, n - k)
        panel = H[:, k:, k:k + kb].clone()
        Vp, taup = torch.geqrf(panel)
        H[:, k:, k:k + kb] = Vp
        tau[:, k:k + kb] = taup
        if k + kb < n:
            V = torch.tril(Vp, -1)
            ar = torch.arange(kb, device=V.device)
            V[:, ar, ar] = 1.0
            T = _build_T(V, taup)
            trailing = H[:, k:, k + kb:]
            W = V.transpose(-2, -1) @ trailing
            TW = T.transpose(-2, -1) @ W
            H[:, k:, k + kb:] = trailing - V @ TW
    return H.float(), tau.float()


def custom_kernel(data: input_t) -> output_t:
    A = data
    squeeze = False
    if A.dim() == 2:
        A = A.unsqueeze(0); squeeze = True
    n = A.shape[-1]
    if n <= SMALL_N:
        H, tau = torch.geqrf(A)
    else:
        H, tau = blocked_hh(A)
    if squeeze:
        return H.squeeze(0), tau.squeeze(0)
    return H, tau
