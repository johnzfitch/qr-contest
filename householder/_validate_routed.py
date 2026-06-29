import sys, pathlib
import torch
HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common"), str(QRPY / "householder")]
from oracle import make_batch, check
import submission_routed as S

STRESS = ["dense", "mixed", "rankdef", "clustered", "nearrank", "nearcollinear",
          "rowscaled", "banded", "uppertri"]

def run(ns, vb=48):
    worst = 0.0
    allok = True
    for n in ns:
        bad = []
        for case in STRESS:
            A = make_batch(vb, n, 2, case, seed=0).to("cuda").contiguous()
            H, tau = S.custom_kernel(A)
            fr, og, ft, ot, ps = check(A, H, tau)
            worst = max(worst, fr / ft, og / ot)
            if not ps.all():
                bad.append(f"{case}({int(ps.sum())}/{vb})")
        route = "geqrf" if n >= 1536 else ("fused" if n == 512 else "wave")
        status = "OK" if not bad else "FAIL: " + ",".join(bad)
        if bad: allok = False
        print(f"  n={n:<5d} [{route:5s}] {status}")
    print(f"  worst margin = {worst:.4f}  (<1.0 == pass)")
    return allok and worst < 1.0

if __name__ == "__main__":
    assert torch.cuda.is_available()
    ok = run((176, 352, 512, 1024, 2048))
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
