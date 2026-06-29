"""Blocked WY-Householder QR — the candidate NEW spine (replaces CholeskyQR + modLU).

Why: blocked Householder produces the geqrf (H, tau) contract DIRECTLY (no Gram, no
Cholesky, no TRSM, no modified-LU reconstruction, no theta-gate, no geqrf fallback). It is
unconditionally backward-stable, so rankdef / clustered / nearrank pass natively — the three
shapes that currently dump to the 80x-slow geqrf fallback (1115 / 423 / 279 ms).

This file validates the MATH on CPU (torch, no CUDA) before the CUDA port. Two mirrors:
  * hh_qr_unblocked : batched geqr2 (LAPACK DLARFG column loop) -> reference (H, tau).
  * hh_qr_blocked   : panel geqr2 + DLARFT T-factor + WY trailing update (I - V Tᵀ Vᵀ)C.
                      This is the exact arithmetic the CUDA kernel will run (BLAS-3 trailing).
Both must pass oracle.check on all 12 shapes, INCLUDING the stress cases, with no fallback.

LAPACK conventions (so torch.linalg.householder_product(H, tau) reconstructs Q):
  DLARFG on [alpha; x]:  xnorm = ||x|| (tail below diagonal)
    xnorm == 0 -> tau = 0, beta = alpha, v = 0          (already triangular, no reflection)
    else       -> beta = -sign(alpha)*||[alpha;x]||,  tau = (beta - alpha)/beta,
                  v = x/(alpha - beta)   (implicit leading 1);  R[j,j] = beta
  Apply H = I - tau [1;v][1;v]ᵀ to trailing columns.

Run:  python3 dev/kqr_hh.py
"""
import torch
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check, SHAPES

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Unblocked batched Householder QR (geqr2) -> reference (H, tau).              #
# --------------------------------------------------------------------------- #
def hh_qr_unblocked(A):
    b, m, ncols = A.shape                      # rectangular panel: m rows, ncols cols (m >= ncols)
    R = A.clone()
    tau = torch.zeros(b, ncols, dtype=A.dtype, device=A.device)
    for j in range(ncols):
        x = R[:, j:, j]                        # (b, m-j) column from diagonal down
        alpha = x[:, 0].clone()                # MUST clone — x[:,0] aliases R[:,j,j]
        if x.shape[1] > 1:
            xnorm = x[:, 1:].norm(dim=1)       # tail norm (below diagonal)
        else:
            xnorm = torch.zeros_like(alpha)
        refl = xnorm > 0
        normfull = torch.sqrt(alpha * alpha + xnorm * xnorm)
        sign = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        beta = torch.where(refl, -sign * normfull, alpha)
        tau_j = torch.where(refl, (beta - alpha) / torch.where(refl, beta, torch.ones_like(beta)),
                            torch.zeros_like(alpha))
        denom = torch.where(refl, alpha - beta, torch.ones_like(alpha))
        R[:, j, j] = beta
        if x.shape[1] > 1:
            v = torch.where(refl.unsqueeze(1), x[:, 1:] / denom.unsqueeze(1),
                            torch.zeros_like(x[:, 1:]))
            R[:, j + 1:, j] = v
        if j < ncols - 1:
            # apply H_j to trailing columns R[:, j:, j+1:]
            mj = m - j
            vfull = torch.ones(b, mj, dtype=A.dtype, device=A.device)
            if mj > 1:
                vfull[:, 1:] = R[:, j + 1:, j]
            C = R[:, j:, j + 1:]               # (b, mj, c)
            vtC = torch.einsum('bm,bmc->bc', vfull, C)
            R[:, j:, j + 1:] = C - tau_j.view(b, 1, 1) * torch.einsum('bm,bc->bmc', vfull, vtC)
        tau[:, j] = tau_j
    return R, tau                              # below diag = reflectors, on/above = R


# --------------------------------------------------------------------------- #
# DLARFT: build the nb x nb upper-triangular WY T from panel reflectors V,tau. #
#   product H_0..H_{nb-1} = I - V T Vᵀ ;  V is (b, m, nb) unit-lower.          #
# --------------------------------------------------------------------------- #
def dlarft(V, tau):
    b, m, nb = V.shape
    T = torch.zeros(b, nb, nb, dtype=V.dtype, device=V.device)
    for j in range(nb):
        T[:, j, j] = tau[:, j]
        if j > 0:
            t = -tau[:, j:j + 1] * torch.einsum('bmi,bm->bi', V[:, :, :j], V[:, :, j])  # (b,j)
            T[:, :j, j] = torch.einsum('bik,bk->bi', T[:, :j, :j], t)
    return T


# --------------------------------------------------------------------------- #
# Blocked WY-Householder QR — mirrors the CUDA kernel (panel geqr2 + WY apply).#
# --------------------------------------------------------------------------- #
def hh_qr_blocked(A, nb=64):
    b, n, _ = A.shape
    R = A.clone()
    tau = torch.zeros(b, n, dtype=A.dtype, device=A.device)
    for p0 in range(0, n, nb):
        pe = min(p0 + nb, n)
        w = pe - p0
        # (1) panel factor: unblocked geqr2 on R[:, p0:, p0:pe] -> reflectors + R + tau
        Hp, taup = hh_qr_unblocked(R[:, p0:, p0:pe].clone())
        R[:, p0:, p0:pe] = Hp
        tau[:, p0:pe] = taup
        if pe < n:
            # (2) form V (unit-lower, m x w) and T (w x w)
            m = n - p0
            V = torch.tril(R[:, p0:, p0:pe], -1)
            diag = torch.zeros(b, m, w, dtype=A.dtype, device=A.device)
            idx = torch.arange(w, device=A.device)
            diag[:, idx, idx] = 1.0
            V = V + diag
            T = dlarft(V, taup)
            # (3) WY trailing update: C -= V Tᵀ Vᵀ C,  C = R[:, p0:, pe:]
            C = R[:, p0:, pe:]
            VtC = torch.einsum('bmi,bmc->bic', V, C)
            TtVtC = torch.einsum('bki,bkc->bic', T, VtC)     # Tᵀ @ VtC
            R[:, p0:, pe:] = C - torch.einsum('bmi,bic->bmc', V, TtVtC)
    return R, tau


# --------------------------------------------------------------------------- #
# Validation over the 12 benchmark shapes (small batch on CPU; stress kept).   #
# --------------------------------------------------------------------------- #
def _report(tag, A, H, tau):
    fr, og, ft, ot, ps = check(A, H, tau)
    flag = "OK  " if ps.all() else "FAIL"
    print(f"  {tag:26s} factor={fr:.2e}/{ft:.2e}  ortho={og:.2e}/{ot:.2e}  "
          f"{flag} ({int(ps.sum())}/{len(ps)})")
    return bool(ps.all())


def main():
    print(f"device={DEV}  blocked WY-Householder QR — algorithm validation\n")
    # CPU: shrink batch, cap n at 1024 (skip 2048/4096 dense — same code path as 1024 dense).
    cpu_shapes = []
    for (b, n, cond, case) in SHAPES:
        if n > 1024:
            continue
        bb = 4 if n <= 512 else 2
        cpu_shapes.append((bb, n, cond, case))

    allok = True
    print("--- unblocked geqr2 (reference) ---")
    for (b, n, cond, case) in cpu_shapes:
        A = make_batch(b, n, cond, case, seed=0)
        H, tau = hh_qr_unblocked(A)
        allok &= _report(f"b{b}_n{n}_{case}", A, H, tau)

    print("\n--- blocked WY (nb=64) — the CUDA arithmetic ---")
    for (b, n, cond, case) in cpu_shapes:
        A = make_batch(b, n, cond, case, seed=0)
        H, tau = hh_qr_blocked(A, nb=64)
        allok &= _report(f"b{b}_n{n}_{case}", A, H, tau)

    print(f"\nALL PASS: {allok}")


if __name__ == "__main__":
    main()
