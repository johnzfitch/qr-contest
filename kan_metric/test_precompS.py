"""Does a PRECOMPUTED sign matrix S (vs. interleaved per-step) still give a valid
reconstruction? If yes, large-n reconstruction becomes a plain no-pivot LU (simple
blocked kernel: panel-factor + trailing GEMM)."""
import torch
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check
from kqr import robust_cqr

torch.backends.cuda.matmul.allow_tf32 = False


def recon_precomp_S(Q, R_chol):
    n = Q.shape[-1]
    d = Q.diagonal(dim1=-2, dim2=-1)
    S = torch.where(d > 0, -torch.ones_like(d), torch.ones_like(d))
    B = Q - torch.diag_embed(S)
    P, L, U = torch.linalg.lu(B, pivot=False)            # no-pivot LU; P empty (identity)
    V = torch.tril(L, -1)
    tau = 2.0 / (V.pow(2).sum(1) + 1.0)
    R = S.unsqueeze(-1) * R_chol
    return V + torch.triu(R), tau


for (b, n, cond, case) in [(40, 176, 2, "dense"), (40, 352, 2, "dense"),
                           (64, 512, 2, "dense"), (40, 512, 0, "clustered"),
                           (40, 512, 0, "rankdef"), (20, 1024, 2, "dense")]:
    A = make_batch(b, n, cond, case, seed=0)
    Q, R, ok = robust_cqr(A, passes=2)
    H, tau = recon_precomp_S(Q, R)
    fr, og, ft, ot, ps = check(A, H, tau)
    flag = "OK" if ps.all() else "FAIL"
    print(f"{case:9s} n={n:4d}: factor={fr:.2e}/{ft:.2e} ortho={og:.2e}/{ot:.2e} "
          f"pass={int(ps.sum())}/{len(ps)} {flag}")
