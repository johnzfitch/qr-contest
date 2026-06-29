"""CPU de-risk of the cluster_qr kernel's EXACT numerics (no GPU needed).

The cluster kernel (kqr_cluster_kan.py cluster_qr_kernel) bets that a tensor-core
formulation clears the contest gate on the hard shapes:
    A --bf16x6 WMMA Gram--> M=A^T A --floored chol--> R --trsm--> Q   (CQR2: x2)
      --modified-LU(Q)--> L,S,tau --factor closure R=S.(Q^T A)--> (H,tau)
Two numerical risks vs a plain fp32 Householder:
  (1) the Gram is bf16x6 (3 bf16 limbs, 6 cross-terms) -- NOT fp32;
  (2) CQR2 squares kappa and the kernel's chol has NO Tikhonov shift, only a
      diagonal floor (s>1e-30 ? sqrt(s) : 1e-15) -- exactly chol_blk.
This mirrors BOTH faithfully in torch and runs oracle.check on the contest's
degenerate cases at the cluster's target n (256/352/512). If this passes on CPU,
the only thing left for the kernel is the cluster/DSMEM mechanics (needs B200).

Run:  /Users/Zack/Projects/qr-contest/.venv/bin/python kan_metric/cluster_mirror_derisk.py
"""
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
import torch
from oracle import make_batch, check, EPS32
from irregular.exp1_irregular import modlu_mirror, robust_cqr  # reuse validated mirrors

torch.manual_seed(0)
DEV = "cpu"


# ---- bf16x6 Gram: 3 limbs per operand, the kernel's exact 6-term order -------
def _limbs3(X):
    h = X.to(torch.bfloat16).to(torch.float32)
    r1 = X - h
    m = r1.to(torch.bfloat16).to(torch.float32)
    r2 = r1 - m
    l = r2.to(torch.bfloat16).to(torch.float32)
    return h, m, l


def gram_bf16x6(A):
    """M = A^T A via 6 bf16-input / fp32-accumulate products (kernel order)."""
    ah, am, al = _limbs3(A)
    bh, bm, bl = ah, am, al              # Gram: both operands are A
    M = (ah.mT @ bh) + (ah.mT @ bm) + (am.mT @ bh)
    M = M + (ah.mT @ bl) + (al.mT @ bh) + (am.mT @ bm)
    return M


def gram_fp32(A):
    return A.mT @ A


# ---- floored upper Cholesky: R^T R = M, mirrors chol_blk exactly -------------
def chol_floor_upper(M):
    b, n, _ = M.shape
    R = torch.zeros_like(M)
    for j in range(n):
        s = M[:, j, j] - (R[:, :j, j] ** 2).sum(1)
        rjj = torch.where(s > 1e-30, s.clamp_min(1e-30).sqrt(), torch.full_like(s, 1e-15))
        R[:, j, j] = rjj
        if j + 1 < n:
            s2 = M[:, j, j + 1:] - torch.einsum('bk,bkc->bc', R[:, :j, j], R[:, :j, j + 1:])
            R[:, j, j + 1:] = s2 / rjj.unsqueeze(-1)
    return R


def cqr2(A, gram, chol):
    """Two CholeskyQR passes with the given Gram + Cholesky; returns Q2."""
    R1 = chol(gram(A))
    Q1 = torch.linalg.solve_triangular(R1, A, upper=True, left=False)
    R2 = chol(gram(Q1))
    Q2 = torch.linalg.solve_triangular(R2, Q1, upper=True, left=False)
    return Q2


def recon_full(A, Q):
    V, tau, S = modlu_mirror(Q)
    R = S.unsqueeze(-1) * (Q.mT @ A)
    H = torch.tril(V, -1) + torch.triu(R)
    return H, tau


# ---- the three variants under test -------------------------------------------
def kernel_faithful(A):                  # what cluster_qr ACTUALLY computes
    return recon_full(A, cqr2(A, gram_bf16x6, chol_floor_upper))


def fp32_floor(A):                       # isolate the bf16-Gram effect
    return recon_full(A, cqr2(A, gram_fp32, chol_floor_upper))


def fp32_shifted(A):                     # would a Tikhonov shift rescue failures?
    Q, _, _ = robust_cqr(A, passes=2)
    return recon_full(A, Q)


def geqrf_control(A):                    # true Householder (the proven 19/19 path)
    return torch.geqrf(A)


VARIANTS = [("kernel bf16x6+floor", kernel_faithful),
            ("fp32 Gram+floor    ", fp32_floor),
            ("fp32 shifted-CQR2   ", fp32_shifted),
            ("geqrf control       ", geqrf_control)]

CASES = ["dense", "mixed", "rankdef", "clustered", "nearrank"]
SHAPES = [(256, 16), (352, 16), (512, 12)]   # (n, batch) -- cluster targets


def run():
    print(f"factor_tol/ortho_tol shown as multiples; pass = factor_res<ftol & ortho<otol")
    for n, b in SHAPES:
        ftol, otol = 20 * n * EPS32, 100 * n * EPS32
        print(f"\n===== n={n}  batch={b}   (ftol={ftol:.2e}  otol={otol:.2e}) =====")
        print(f"{'case':10s} | " + " | ".join(f"{name}" for name, _ in VARIANTS))
        for case in CASES:
            A = make_batch(b, n, 2, case, seed=0).to(DEV).float()
            cells = []
            for name, fn in VARIANTS:
                try:
                    H, tau = fn(A)
                    fr, og, ft, ot, ps = check(A, H, tau)
                    npass = int(ps.sum())
                    flag = "OK " if ps.all() else "XX "
                    cells.append(f"{flag}{npass}/{b} f={fr/ft:5.2f} o={og/ot:4.2f}")
                except Exception as e:
                    cells.append(f"ERR {type(e).__name__}")
            print(f"{case:10s} | " + " | ".join(cells))


if __name__ == "__main__":
    run()
