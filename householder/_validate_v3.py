import sys, pathlib, torch
QRPY = pathlib.Path("/workspace/qr")
sys.path[:0] = [str(QRPY), str(QRPY / "common"), str(QRPY / "householder")]
from oracle import make_batch, check
import submission_routed_v3 as S

print("GEQRF_N =", S.GEQRF_N)
STRESS = ["dense", "mixed", "rankdef", "clustered", "nearrank",
          "nearcollinear", "rowscaled", "banded", "uppertri"]
worst = 0.0
allok = True
for n in (176, 352, 512, 1024, 2048, 4096):
    route = "geqrf" if n >= S.GEQRF_N else ("fused" if n >= 512 else "wave")
    vb = 48 if n <= 1024 else (8 if n == 2048 else 2)
    bad = []
    for case in STRESS:
        A = make_batch(vb, n, 2, case, seed=0).cuda().contiguous()
        H, tau = S.custom_kernel(A)
        fr, og, ft, ot, ps = check(A, H, tau)
        worst = max(worst, fr / ft, og / ot)
        if not ps.all():
            bad.append(case)
    if bad:
        allok = False
    status = "OK" if not bad else "FAIL: " + ",".join(bad)
    print("n={:<5d} [{:5s}] {}".format(n, route, status))
print("worst margin = {:.4f}".format(worst))
print("RESULT:", "PASS" if allok and worst < 1.0 else "FAIL")
