"""Shifted-CholeskyQR3 ACCURACY de-risk (speed irrelevant — uses torch chol/solve).

The CQR speed path is scoped (custom tensor-core chol+tri-inverse). The UNKNOWN
risk is accuracy on the 5 rank-deficient/ill-conditioned board shapes (cond^2
amplification). This probe asks ONLY: does Shifted-CQR3 pass the oracle gate on
all 12 shapes, at fp32 (algorithm ceiling) and tf32-GEMM (realistic kernel precision)?

Gate (mirrors oracle.check, evaluated on raw CQR Q,R; modLU recon is exact to 1e-14):
  factor = ||R - Q^T A||_1 / ||A||_1   < 20 n eps32
  ortho  = ||Q^T Q - I||_1             < 100 n eps32
margin = max(factor/ftol, ortho/otol); pass = margin < 1.

  source /workspace/qr/env.sh && python householder/shifted_cqr3_accuracy_derisk.py
"""
import sys, pathlib
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, SHAPES                          # noqa: E402

EPS32 = torch.finfo(torch.float32).eps


def _m1(X):  # max over batch of matrix 1-norm (max abs column sum)
    return X.abs().sum(dim=-2).amax(dim=-1)


def chol_qr(A, M):
    """one CholeskyQR step given Gram M ~ A^T A (already shifted if desired): R=chol(M)^T upper, Q=A R^-1."""
    L, info = torch.linalg.cholesky_ex(M)
    R = L.transpose(-1, -2)  # upper
    Q = torch.linalg.solve_triangular(R, A, upper=True, left=False)
    return Q, R, info


def shifted_cqr3(A, tf32, dtype=torch.float32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    Aw = A.to(dtype)
    n = A.shape[-1]
    u = torch.finfo(dtype).eps
    M = torch.matmul(Aw.transpose(1, 2), Aw)
    # Fukaya-style shift s = 11*(2n^2 + n)*u*||A||_2^2 ; ||A||_2^2=||M||_2 <= ||M||_1 (symmetric)
    Mnorm = M.abs().sum(dim=-2).amax(dim=-1).clamp_min(1e-300)  # ||M||_1 upper-bounds ||M||_2
    s = (11.0 * (2 * n * n + n) * u) * Mnorm
    Ms = M + s.view(-1, 1, 1) * torch.eye(n, device=A.device, dtype=dtype).unsqueeze(0)
    Q1, R1, _ = chol_qr(Aw, Ms)
    M2 = torch.matmul(Q1.transpose(1, 2), Q1)
    Q2, R2, _ = chol_qr(Q1, M2)
    M3 = torch.matmul(Q2.transpose(1, 2), Q2)
    Q3, R3, _ = chol_qr(Q2, M3)
    R = torch.matmul(torch.matmul(R3, R2), R1)
    torch.backends.cuda.matmul.allow_tf32 = False
    return Q3, R


def gate(A, Q, R):
    Ad = A.double(); Qd = Q.double(); Rd = R.double()
    n = A.shape[-1]
    factor = (_m1(Rd - Qd.transpose(1, 2) @ Ad) / _m1(Ad))
    I = torch.eye(n, device=A.device, dtype=torch.float64).unsqueeze(0)
    ortho = _m1(Qd.transpose(1, 2) @ Qd - I)
    ftol, otol = 20 * n * EPS32, 100 * n * EPS32
    margin = torch.maximum(factor / ftol, ortho / otol)
    return margin.amax().item(), (margin < 1.0).all().item()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("== Shifted-CQR3 accuracy on all 12 board shapes (gate margin <1 = pass) ==")
    print(f"{'shape':22s} {'fp64 marg':>10s} {'64':>3s} {'fp32 marg':>10s} {'32':>3s} {'tf32 marg':>10s} {'tf':>3s}")
    fails = {"fp64": [], "fp32": [], "tf32": []}
    for b, n, cond, case in SHAPES:
        A = make_batch(b, n, cond, case, seed=0).cuda().contiguous()
        Qd, Rd = shifted_cqr3(A, False, torch.float64); md, pd = gate(A, Qd, Rd)
        Qf, Rf = shifted_cqr3(A, False, torch.float32); mf, pf = gate(A, Qf, Rf)
        Qt, Rt = shifted_cqr3(A, True, torch.float32);  mt, pt = gate(A, Qt, Rt)
        for k, p, nc in [("fp64", pd, (n, case)), ("fp32", pf, (n, case)), ("tf32", pt, (n, case))]:
            if not p: fails[k].append(nc)
        print(f"{str((b,n,case)):22s} {md:10.4f} {'OK' if pd else 'XX':>3s} {mf:10.4f} {'OK' if pf else 'XX':>3s} {mt:10.4f} {'OK' if pt else 'XX':>3s}")
    print(f"\n  fp64 (algorithm ceiling) fails: {fails['fp64'] if fails['fp64'] else 'NONE'}")
    print(f"  fp32 fails: {fails['fp32'] if fails['fp32'] else 'NONE'}")
    print(f"  tf32 fails: {fails['tf32'] if fails['tf32'] else 'NONE'}")
    print("  CEILING: fp64 all-pass -> algorithm SOUND (precision is the separable problem); fp64 rankdef FAIL -> algorithm capped.")
