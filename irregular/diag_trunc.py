"""Diagnose the Path-C (leading-r truncation) failures at n=512: is it math fragility
(the leading-r block is itself rank-deficient, so leading columns don't span range(A)),
or a device/implementation effect (passes on CPU, fails on GPU)?

Same pure-torch code as exp1; the only knobs that changed between the prior CPU and GPU
runs were n (64 -> 512) and batch (4 -> 640). This isolates the cause.
"""
import torch
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check, EPS32
from exp1_irregular import recon_byrank, reveal_rank, recon_lu_precond

DEV = "cuda" if torch.cuda.is_available() else "cpu"
b, n = 640, 512
A = make_batch(b, n, 0, "rankdef", seed=0)
r_vec = reveal_rank(A)
r = int(r_vec[0])
H, tau = recon_byrank(A, r_vec)
fr, og, ft, ot, ps = check(A, H, tau)
fail = (~ps).nonzero(as_tuple=True)[0]
passing = ps.nonzero(as_tuple=True)[0]
print(f"[{DEV}] rankdef {b}x{n}: r={r}  Path-C pass={int(ps.sum())}/{b}  fails={fail.tolist()}")

if len(fail) == 0:
    print("  no failures to diagnose."); raise SystemExit

# (1) Is the leading-r block rank-deficient for the failures? -------------------------
sv_lead = torch.linalg.svdvals(A[:, :, :r].double())          # sing. vals of A[:, :, :r]
rank_lead = (sv_lead > (20 * n * EPS32) * sv_lead[:, :1]).sum(-1)
relsmin = (sv_lead[:, -1] / sv_lead[:, 0])
print(f"  leading-{r} block numerical rank:  FAIL = {rank_lead[fail].tolist()}")
print(f"                                      PASS range = [{int(rank_lead[passing].min())}, "
      f"{int(rank_lead[passing].max())}]  (full would be {r})")
print(f"  leading-{r} block rel. smallest sv: FAIL = {[f'{x:.1e}' for x in relsmin[fail].tolist()]}")
print(f"                                      PASS min = {relsmin[passing].min().item():.1e}")

# (2) Re-run the SAME failing matrices on CPU (device-independence test) ---------------
if DEV == "cuda":
    Ac = A[fail].cpu()
    Hc, tc = recon_byrank(Ac, reveal_rank(Ac))
    frc, ogc, ftc, otc, psc = check(Ac, Hc, tc)
    print(f"  SAME {len(fail)} matrices re-run on CPU: Path-C pass={int(psc.sum())}/{len(fail)} "
          f"factor={frc:.2e}  ->  {'DEVICE-SPECIFIC (check CUDA docs)' if int(psc.sum())==len(fail) else 'fails on CPU too => math, not device'}")

# (3) Does LU-precond handle exactly these failures? ----------------------------------
Hl, tl = recon_lu_precond(A[fail])
frl, ogl, ftl, otl, psl = check(A[fail], Hl, tl)
print(f"  LU-precond on the same {len(fail)} matrices: pass={int(psl.sum())}/{len(fail)} factor={frl:.2e}")
