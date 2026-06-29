"""Oracle / correctness foundation for the qr-contest fused kernel.

This is dev scaffolding (NOT submitted). It encodes, in plain PyTorch:
  * the contest data generators (approximate — real ones live in gpu-mode/reference-kernels),
  * the EXACT checker (factor residual + orthogonality, L1, FP64),
  * the reference CholeskyQR pipeline with a per-matrix theta-gate + geqrf fallback,
  * TWO reconstructions of the geqrf (H, tau) contract from a CholeskyQR Q:
       - reconstruct_via_geqrf : ground truth (unambiguously correct)
       - reconstruct_via_lu    : the modified-LU method the CUDA kernel will hand-roll
                                 (Ballard-Demmel-Grigori "reconstruct Householder from Q")

Run on the Runpod B200:  python dev/oracle.py
Everything here must pass before we trust the kernel port.
"""
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False  # keep the reference honest

EPS32 = torch.finfo(torch.float32).eps  # 1.1920929e-07
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# The 12 ranked benchmark shapes (batch, n, cond, case).
SHAPES = [
    (20,   32, 1, "dense"),
    (40,  176, 1, "dense"),
    (40,  352, 1, "dense"),
    (640, 512, 2, "dense"),
    (60, 1024, 2, "dense"),
    (8,  2048, 1, "dense"),
    (2,  4096, 1, "dense"),
    (640, 512, 2, "mixed"),
    (60, 1024, 2, "mixed"),
    (640, 512, 0, "rankdef"),
    (640, 512, 0, "clustered"),
    (60, 1024, 0, "nearrank"),
]


# --------------------------------------------------------------------------- #
# Data generation (approximate; validate the REAL ones via popcorn test mode). #
# --------------------------------------------------------------------------- #
def _randn(b, n, gen):
    return torch.randn(b, n, n, device=DEV, dtype=torch.float32, generator=gen)


def gen_dense(b, n, cond, gen):
    A = _randn(b, n, gen)
    scales = torch.logspace(0, -float(cond), n, device=DEV)  # columns x logspace(0,-cond,n)
    return A * scales.view(1, 1, n)


def _with_singular_values(b, n, svals, gen):
    """Build a batch with prescribed singular values via random orthogonal U, V."""
    U, _ = torch.linalg.qr(_randn(b, n, gen))
    V, _ = torch.linalg.qr(_randn(b, n, gen))
    return (U * svals.view(1, 1, n)) @ V.mT


def gen_rankdef(b, n, gen, rank_frac=0.5):
    r = int(n * rank_frac)
    s = torch.zeros(n, device=DEV)
    s[:r] = torch.logspace(0, -1, r, device=DEV)  # tail is exactly zero -> rank deficient
    return _with_singular_values(b, n, s, gen)


def gen_nearrank(b, n, gen, rank_frac=0.5):
    r = int(n * rank_frac)
    s = torch.empty(n, device=DEV)
    s[:r] = torch.logspace(0, -1, r, device=DEV)
    s[r:] = 1e-8  # tiny but nonzero tail
    return _with_singular_values(b, n, s, gen)


def gen_clustered(b, n, gen):
    s = torch.ones(n, device=DEV)
    s[n // 2:] = 1e-6  # two tight clusters of singular values
    return _with_singular_values(b, n, s, gen)


def gen_mixed(b, n, cond, gen):
    """Heterogeneous batch: well-conditioned majority interleaved with stress structures."""
    A = gen_dense(b, n, cond, gen)
    stress = [gen_rankdef, gen_nearrank, gen_clustered]
    for i in range(b):
        if i % 4 == 3:  # ~25% stress, scattered
            A[i] = stress[(i // 4) % 3](1, n, gen)[0]
    return A


# --------------------------------------------------------------------------- #
# Extra stress structures (Layer 2.2): named in the spec but absent from the   #
# 12 ranked shapes. Legal test-set inputs; all return (b,n,n) FP32 on DEV.     #
# --------------------------------------------------------------------------- #
def gen_banded(b, n, gen, bw=None):
    """Dense, zeroed outside the band |i-j| <= bw (default n//8). Cheaper panels."""
    if bw is None:
        bw = max(1, n // 8)
    A = _randn(b, n, gen)
    ii = torch.arange(n, device=DEV)
    band = (ii.view(n, 1) - ii.view(1, n)).abs() <= bw
    return A * band.view(1, n, n)


def gen_rowscaled(b, n, cond, gen):
    """Dense with a logspace dynamic range applied to ROWS (not columns)."""
    A = _randn(b, n, gen)
    scales = torch.logspace(0, -float(cond), n, device=DEV)
    return A * scales.view(1, n, 1)


def gen_nearcollinear(b, n, gen, eps=1e-7):
    """Dense, but column 1 = column 0 + eps*noise — a specific near-collinear pair
       (distinct from nearrank, which is a whole tail of tiny singular values)."""
    A = _randn(b, n, gen)
    noise = torch.randn(b, n, device=DEV, dtype=torch.float32, generator=gen)
    A[:, :, 1] = A[:, :, 0] + eps * noise
    return A


def gen_uppertri(b, n, gen):
    """Already upper-triangular: R ~ A, reflectors near-trivial (no-op path)."""
    return torch.triu(_randn(b, n, gen))


def make_batch(b, n, cond, case, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    if case == "dense":
        return gen_dense(b, n, cond, gen)
    if case == "mixed":
        return gen_mixed(b, n, cond, gen)
    if case == "rankdef":
        return gen_rankdef(b, n, gen)
    if case == "nearrank":
        return gen_nearrank(b, n, gen)
    if case == "clustered":
        return gen_clustered(b, n, gen)
    if case == "banded":
        return gen_banded(b, n, gen)
    if case == "rowscaled":
        return gen_rowscaled(b, n, cond, gen)
    if case == "nearcollinear":
        return gen_nearcollinear(b, n, gen)
    if case == "uppertri":
        return gen_uppertri(b, n, gen)
    raise ValueError(case)


# --------------------------------------------------------------------------- #
# The EXACT checker (matches the contest: L1, relative, FP64).                 #
# --------------------------------------------------------------------------- #
def _m1(X):  # per-matrix matrix 1-norm (max abs column sum)
    return torch.linalg.matrix_norm(X, ord=1, dim=(-2, -1))


def check(A, H, tau):
    """Return (max_factor_res, max_ortho, factor_tol, ortho_tol, passed_mask)."""
    n = A.shape[-1]
    Ad, Hd, td = A.double(), H.double(), tau.double()
    Q = torch.linalg.householder_product(Hd, td)
    R = torch.triu(Hd)
    QtA = Q.mT @ Ad
    factor_res = _m1(R - QtA) / _m1(Ad)
    I = torch.eye(n, device=A.device, dtype=torch.float64).expand_as(Q)
    ortho = _m1(Q.mT @ Q - I)  # ||I||_1 = 1
    ftol, otol = 20 * n * EPS32, 100 * n * EPS32
    passed = (factor_res < ftol) & (ortho < otol)
    return factor_res.max().item(), ortho.max().item(), ftol, otol, passed


def check_percol(A, H, tau):
    """Stricter PER-COLUMN-relative factor gate (LAPACK-per-column style).

    Returns (max_factor_ratio, max_ortho, passed_mask), ratio<1 == pass.

    The factor residual is normalized per column by ||A[:,j]||_1 instead of by
    the whole-matrix ||A||_1 that check() uses. They AGREE for backward-stable
    Householder, but a quantized trailing update can pass the matrix-norm gate yet
    fail per-column on a tiny clustered/nearrank column (matrix-norm lets a small
    column borrow the big column's budget). That divergence is exactly the regime
    the benchmark stress-weights, so we surface it pre-submission. The real
    reference-kernels checker is not vendored, so we gate on BOTH norms.
    """
    n = A.shape[-1]
    Ad, Hd, td = A.double(), H.double(), tau.double()
    Q = torch.linalg.householder_product(Hd, td)
    R = torch.triu(Hd)
    res = (R - Q.mT @ Ad).abs().sum(dim=-2)               # (b, n) column 1-norms
    acol = Ad.abs().sum(dim=-2).clamp_min(1e-300)         # (b, n) column 1-norms
    ftol, otol = 20 * n * EPS32, 100 * n * EPS32
    ratio = (res / (ftol * acol)).amax(dim=-1)            # (b,) budget fraction used
    I = torch.eye(n, device=A.device, dtype=torch.float64).expand_as(Q)
    ortho = _m1(Q.mT @ Q - I)
    passed = (ratio < 1.0) & (ortho < otol)
    return ratio.max().item(), ortho.max().item(), passed


# --------------------------------------------------------------------------- #
# Reconstruction route 1: ground truth via geqrf(Q).                          #
# --------------------------------------------------------------------------- #
def reconstruct_via_geqrf(Q, A):
    """Reflectors representing Q (up to a sign matrix S), with R = triu(Q_tilde^T A)."""
    Hq, tau = torch.geqrf(Q)                       # prod(reflectors) = Q @ S
    Qt = torch.linalg.householder_product(Hq, tau)  # = Q @ S, exactly orthogonal
    R = torch.triu(Qt.mT @ A)
    H = torch.tril(Hq, diagonal=-1) + R            # vectors below, R on/above diag
    return H, tau


# --------------------------------------------------------------------------- #
# Reconstruction route 2: modified-LU (what the kernel hand-rolls).           #
#   Q - S = L U  (no pivoting); L (unit lower) = Householder vectors V.       #
#   tau set to 2/||v||^2 so householder_product(V, tau) is EXACTLY orthogonal #
#   (orthogonality is free); R = triu(Q_tilde^T A) carries the factor.        #
# --------------------------------------------------------------------------- #
def reconstruct_via_lu(Q, A):
    b, n, _ = Q.shape
    B = Q.clone()
    V = torch.zeros_like(Q)
    idx = torch.arange(b, device=Q.device)
    for i in range(n):
        diag = B[:, i, i]
        si = -torch.sign(diag)
        si = torch.where(si == 0, torch.ones_like(si), si)
        piv = diag - si                       # (Q - S)_{ii} after prior Schur updates
        B[:, i, i] = piv
        V[:, i, i] = 1.0
        if i + 1 < n:
            col = B[:, i + 1:, i] / piv.unsqueeze(-1)         # L below diagonal
            V[:, i + 1:, i] = col
            urow = B[:, i, i + 1:]                            # U row (right of pivot)
            B[:, i + 1:, i + 1:] -= col.unsqueeze(-1) * urow.unsqueeze(1)
    Vv = torch.tril(V, diagonal=-1) + torch.eye(n, device=Q.device).expand_as(V)
    vnorm2 = (Vv * Vv).sum(dim=1)             # ||v_i||^2 including the unit diagonal
    tau = 2.0 / vnorm2
    H_vecs = torch.tril(Vv, diagonal=-1)
    Qt = torch.linalg.householder_product(H_vecs, tau)  # exactly orthogonal by construction
    R = torch.triu(Qt.mT @ A)
    H = H_vecs + R
    return H, tau


# --------------------------------------------------------------------------- #
# CholeskyQR pipeline with theta-gate + per-matrix geqrf fallback.            #
# --------------------------------------------------------------------------- #
def cholesky_qr(A, passes=1):
    """Return Q, R (A = Q R) via (possibly repeated) CholeskyQR, plus per-matrix ok mask."""
    M = A.mT @ A
    L, info = torch.linalg.cholesky_ex(M)
    R = L.mT
    ok = info == 0
    # solve_triangular needs nonsingular R; patch failed ones to identity to avoid NaNs.
    Rs = torch.where(ok.view(-1, 1, 1), R, torch.eye(A.shape[-1], device=A.device))
    Q = torch.linalg.solve_triangular(Rs, A, upper=True, left=False)
    if passes > 1:
        Q2, R2, ok2 = cholesky_qr(Q, passes - 1)
        Q, R, ok = Q2, R2 @ R, ok & ok2
    return Q, R, ok


def pipeline(A, recon=reconstruct_via_lu, passes=2, gate_mult=1.0):
    """Fast path = CholeskyQR2 + reconstruction; theta-gate routes failures to geqrf."""
    n = A.shape[-1]
    Q, R, ok = cholesky_qr(A, passes=passes)
    # theta-gate: orthogonality defect of the cheap path (FP32) vs. the contest ortho tol.
    I = torch.eye(n, device=A.device).expand_as(Q)
    defect = _m1((Q.mT @ Q - I).float())
    good = ok & (defect < gate_mult * 100 * n * EPS32)
    H, tau = recon(Q, A)
    # Fallback: honest FP32 Householder for the gated-out matrices.
    bad = ~good
    if bad.any():
        Hb, taub = torch.geqrf(A[bad])
        H[bad], tau[bad] = Hb, taub
    return H, tau, good


# --------------------------------------------------------------------------- #
# Test driver.                                                                 #
# --------------------------------------------------------------------------- #
def _bench(fn, iters=3):
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def main():
    print(f"device={DEV}  eps32={EPS32:.3e}\n")

    # (1) Validate the two reconstructions agree, on a clean well-conditioned batch.
    print("--- reconstruction cross-check (small, well-conditioned) ---")
    A = make_batch(8, 64, 1, "dense", seed=1)
    Q, R, _ = cholesky_qr(A, passes=2)
    for name, recon in [("geqrf", reconstruct_via_geqrf), ("lu", reconstruct_via_lu)]:
        H, tau = recon(Q, A)
        fr, og, ft, ot, ps = check(A, H, tau)
        print(f"  {name:5s}: factor={fr:.2e}/{ft:.2e}  ortho={og:.2e}/{ot:.2e}  "
              f"pass={int(ps.sum())}/{len(ps)}")

    # (2) Full pipeline over every benchmark shape.
    print("\n--- full pipeline over benchmark shapes ---")
    print(f"  {'shape':28s} {'factor(res/tol)':22s} {'ortho(res/tol)':22s} pass   t(us)")
    for (b, n, cond, case) in SHAPES:
        A = make_batch(b, n, cond, case, seed=0)
        H, tau, good = pipeline(A)
        fr, og, ft, ot, ps = check(A, H, tau)
        t = _bench(lambda: pipeline(A)) if n <= 1024 else float("nan")
        tag = f"b{b}_n{n}_c{cond}_{case}"
        ok = "OK " if ps.all() else "FAIL"
        print(f"  {tag:28s} {fr:.2e}/{ft:.2e}    {og:.2e}/{ot:.2e}    {ok} "
              f"({int(ps.sum())}/{len(ps)})  {t:8.0f}")


if __name__ == "__main__":
    main()
