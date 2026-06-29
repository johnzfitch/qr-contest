"""Exp 1 — does the EXISTING reflection-free spine already carry the irregular cell?

Question this answers (decisively, with numbers, on the real B200 shapes):
  Path A  shifted-CholeskyQR2 (robust_cqr) + full square modlu + factor-closure R=S.(Q^T A),
          NO geqrf fallback.  -> per-matrix pass rate at the contest gate on rankdef / nearrank
          / clustered / mixed.  If this already passes, the "rank-reveal mechanism" debate
          (pivoted-Cholesky vs randomized sketch) is moot for correctness.
  Path B  the breakthrough's optimization: zero-pad tau at the sub-threshold CholeskyQR pivots
          (reconstruct only the r significant reflectors). Same Q, fewer reflectors. Confirms
          the rank-truncation is correct AND tells us the numerical rank r per shape.

Also reports, per shape: the true numerical rank (from svdvals), the rank Path B detects, and
whether clustered folds into the truncation path (the claim that gen_clustered's 1e-6 tail is
sub-tolerance, so NO separate LU-precondition cell is needed).

Reflection-free throughout: robust_cqr is Gram->chol->TRSM, modlu is the closed-form BDG
reconstruction. No Householder reflections applied anywhere. geqrf is used ONLY as an
independent oracle to report "how far is the cheap path from the reference", never in a result.

Run on the Runpod B200:  python dev/exp1_irregular.py
"""
import torch
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check, EPS32

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# robust_cqr inlined verbatim from kqr.py (pure torch; importing kqr would     #
# trigger its module-level CUDA load_inline, which fails on a CPU-only host).  #
# --------------------------------------------------------------------------- #
def _chol_with_shift(M):
    n = M.shape[-1]
    L, info = torch.linalg.cholesky_ex(M)
    bad = info != 0
    if bad.any():
        dmax = M.diagonal(dim1=-2, dim2=-1).amax(-1)
        shift = (11.0 * n * EPS32) * dmax.clamp_min(1e-30)
        I = torch.eye(n, device=M.device)
        Mb = M[bad] + shift[bad].view(-1, 1, 1) * I
        Lb, infob = torch.linalg.cholesky_ex(Mb)
        L = L.clone(); L[bad] = Lb
        info = info.clone(); info[bad] = infob
    return L.mT, info == 0


def robust_cqr(A, passes=2):
    """Shifted CholeskyQR{passes}: returns Q, accumulated R (A=QR), per-matrix ok mask."""
    M = A.mT @ A
    R, ok = _chol_with_shift(M)
    Q = torch.linalg.solve_triangular(R, A, upper=True, left=False)
    for _ in range(passes - 1):
        M2 = Q.mT @ Q
        R2, ok2 = _chol_with_shift(M2)
        Q = torch.linalg.solve_triangular(R2, Q, upper=True, left=False)
        R = R2 @ R
        ok = ok & ok2
    return Q, R, ok

# The real stress shapes (batch, n) — exactly the contest's degenerate cases.
STRESS = [
    ("rankdef",   640,  512),
    ("clustered", 640,  512),
    ("nearrank",   60, 1024),
    ("mixed",     640,  512),
    ("mixed",      60, 1024),
]


def modlu_mirror(Q):
    """Pure-torch modified-LU of (Q - S) = L U. Mirrors the CUDA modlu kernel exactly.
    Returns V (strictly-lower Householder vecs, unit diag implicit), tau (2/||v||^2), S (signs)."""
    b, n, _ = Q.shape
    B = Q.clone()
    V = torch.zeros_like(Q)
    S = torch.empty(b, n, device=Q.device)
    for i in range(n):
        d = B[:, i, i]
        si = torch.where(d > 0, -torch.ones_like(d), torch.ones_like(d))  # -sign(d); 0 -> +1
        piv = d - si
        B[:, i, i] = piv
        S[:, i] = si
        V[:, i, i] = 1.0
        if i + 1 < n:
            col = B[:, i + 1:, i] / piv.unsqueeze(-1)
            V[:, i + 1:, i] = col
            B[:, i + 1:, i + 1:] -= col.unsqueeze(-1) * B[:, i, i + 1:].unsqueeze(1)
    tau = 2.0 / (torch.tril(V, -1).pow(2).sum(1) + 1.0)
    return V, tau, S


def recon_full(A, Q):
    """Path A: full square reconstruction with factor closure R = S .* (Q^T A)."""
    V, tau, S = modlu_mirror(Q)
    R = S.unsqueeze(-1) * (Q.mT @ A)
    H = torch.tril(V, -1) + torch.triu(R)
    return H, tau


def modlu_rect(Q1):
    """Rectangular modified-LU: LU of (Q1 - S) over the r columns of an n x r matrix.
    Returns V (n x r, unit diag implicit) and tau (r). The r<n generalization of modlu."""
    b, n, r = Q1.shape
    B = Q1.clone()
    V = torch.zeros_like(Q1)
    for i in range(r):
        d = B[:, i, i]
        si = torch.where(d > 0, -torch.ones_like(d), torch.ones_like(d))
        piv = d - si
        B[:, i, i] = piv
        V[:, i, i] = 1.0
        if i + 1 < n:
            col = B[:, i + 1:, i] / piv.unsqueeze(-1)
            V[:, i + 1:, i] = col
            if i + 1 < r:
                B[:, i + 1:, i + 1:] -= col.unsqueeze(-1) * B[:, i, i + 1:].unsqueeze(1)
    tau = 2.0 / (torch.tril(V, -1).pow(2).sum(1) + 1.0)          # (b, r)
    return V, tau


def recon_rangetrunc(A, r):
    """Path C (single uniform rank r): clean CholeskyQR2 of the LEADING r columns gives the
    range basis Q1 (n x r) in natural order (so Q1^T A is upper-triangular), then rectangular
    modlu + zero-pad tau. NO global shift -- the leading r-block is well-conditioned. The
    trailing n-r identity reflectors complete range(A)^perp for free. Reflection-free."""
    b, n, _ = A.shape
    A1 = A[:, :, :r].contiguous()
    Q1, _, _ = robust_cqr(A1, passes=2)        # leading-r CholeskyQR2 (cholesky_ex + shift guard)
    V, tau_r = modlu_rect(Q1)
    Vn = torch.zeros(b, n, n, device=A.device, dtype=A.dtype)
    Vn[:, :, :r] = torch.tril(V, -1)
    tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    tau[:, :r] = tau_r
    Qh = torch.linalg.householder_product(Vn, tau)              # product of r reflectors
    H = Vn + torch.triu(Qh.mT @ A)                              # factor closure R = triu(Q^T A)
    return H, tau


def recon_byrank(A, r_vec):
    """Path C over a heterogeneous batch: bucket by unique rank, run recon_rangetrunc per
    bucket, scatter back. One bucket for a homogeneous stress batch; a few for `mixed`."""
    b, n, _ = A.shape
    H = torch.zeros(b, n, n, device=A.device, dtype=A.dtype)
    tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    for r in torch.unique(r_vec).tolist():
        idx = (r_vec == r).nonzero(as_tuple=True)[0]
        Hr, tr = recon_rangetrunc(A[idx], int(r))
        H[idx] = Hr
        tau[idx] = tr
    return H, tau


def reveal_rank(A):
    """Per-matrix numerical rank at the contest gate scale (svdvals oracle for Exp 1; the
    kernel will use pivoted-Cholesky / sketch -- this pins the ground-truth r to compare to)."""
    n = A.shape[-1]
    sv = torch.linalg.svdvals(A.double())
    return (sv > (20 * n * EPS32) * sv[:, :1]).sum(-1)


def recon_from_Q(A, Q):
    """modlu reconstruction from any orthonormal Q, with factor closure R = triu(Q_hat^T A)."""
    V, tau, _ = modlu_mirror(Q)
    Vv = torch.tril(V, -1)
    Qh = torch.linalg.householder_product(Vv, tau)
    return Vv + torch.triu(Qh.mT @ A), tau


def recon_lu_precond(A):
    """THE UNIVERSAL irregular path. Partial-pivot LU A = P L U preconditions: L is unit-lower
    with |multipliers| <= 1 (well-conditioned regardless of cond(A)), U carries the conditioning
    and just multiplies into R. CholeskyQR2 on L -> Q_A = P Q_L -> modlu. Row pivoting preserves
    A's natural COLUMN order, so the triangularity gate holds. Passes dense/rankdef/nearrank/
    clustered/mixed uniformly (Exp 1, FP64 gate) with no rank-reveal, truncation, or bucketing.
    Reflection-free. Reserved for the stress cells -- the dense hot path stays pure CholeskyQR2
    so the metric GEMM (not a sequential LU) is the peak operation."""
    P, L, U = torch.linalg.lu(A)
    Q_L, _, _ = robust_cqr(L, passes=2)
    return recon_from_Q(A, P @ Q_L)


def _summ(tag, A, H, tau):
    fr, og, ft, ot, ps = check(A, H, tau)
    flag = "OK  " if ps.all() else "FAIL"
    print(f"    {tag:14s} factor={fr:.2e}/{ft:.2e}  ortho={og:.2e}/{ot:.2e}  "
          f"pass={int(ps.sum()):4d}/{len(ps):<4d} {flag}")
    return int(ps.sum()), len(ps)


def main():
    print(f"device={DEV}  eps32={EPS32:.3e}")
    print("Exp 1 (FP64 gate) -- the irregular cell at the real contest shapes:")
    print("  A existing   = shifted-CQR2 + full modlu (current spine), no geqrf fallback")
    print("  U lu-precond = partial-pivot LU -> CQR2 on L -> modlu  (THE universal path)")
    print("  C truncate   = leading-r CQR2 + rectangular modlu + zero-pad (low-rank perf option)\n")
    for case, b, n in [("dense", 640, 512)] + STRESS:
        A = make_batch(b, n, 2 if case in ("mixed", "dense") else 0, case, seed=0)
        r_vec = reveal_rank(A)
        print(f"  {case:9s} b{b} n{n}:  numerical rank r in "
              f"[{int(r_vec.min())}, {int(r_vec.max())}]  (n={n})")
        _summ("A existing  ", A, *recon_full(A, robust_cqr(A, passes=2)[0]))
        _summ("U lu-precond", A, *recon_lu_precond(A))
        if case in ("rankdef", "nearrank"):          # truncation valid only where the tail ~ 0
            _summ("C truncate  ", A, *recon_byrank(A, r_vec))
        print()


if __name__ == "__main__":
    main()
