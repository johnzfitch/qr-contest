"""tf32 margin seed-variance probe -- the contest draws its OWN inputs, so worst
margin over seeds is the real DQ exposure, not the seed-0 margin.

For the shapes that carry tf32's risk, sweep seeds 0..N and report the WORST tf32
margin. If n512's worst stays comfortably <1.0 across seeds, gating tf32 to n>=512
is safe; if it spikes near/over 1.0, restrict tf32 to n>=1024 (margins <=0.33).

  source /workspace/qr/env.sh && python householder/tf32_seedvar_derisk.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check                          # noqa: E402

spec = importlib.util.spec_from_file_location("jerry", QRPY / "householder" / "jerry_owes_me_lunch.py")
jerry = importlib.util.module_from_spec(spec); spec.loader.exec_module(jerry)

PROBES = [(40, 176, 1, "dense"), (40, 352, 1, "dense"),
          (640, 512, 2, "mixed"), (640, 512, 0, "rankdef"), (640, 512, 0, "clustered"),
          (60, 1024, 2, "mixed"), (60, 1024, 0, "nearrank")]

if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    NSEED = 8
    print(f"== tf32 margin seed-variance (seeds 0..{NSEED-1}) -- worst is the DQ exposure ==")
    print(f"{'shape':22s} {'min marg':>9s} {'max marg':>9s} {'any fail':>9s}")
    for b, n, cond, case in PROBES:
        mn, mx, anyfail = 1e9, 0.0, False
        for s in range(NSEED):
            A = make_batch(b, n, cond, case, seed=s).to("cuda").contiguous()
            H, tau = jerry.custom_kernel(A)
            fr, og, ft, ot, ps = check(A, H, tau)
            m = max(fr / ft, og / ot)
            mn = min(mn, m); mx = max(mx, m); anyfail = anyfail or (not bool(ps.all()))
        flag = "  <-- FAIL" if (anyfail or mx >= 1.0) else ("  thin" if mx > 0.85 else "")
        print(f"{str((b,n,case)):22s} {mn:9.4f} {mx:9.4f} {str(anyfail):>9s}{flag}")
    torch.backends.cuda.matmul.allow_tf32 = False
    print("\n  n>=512 safe to tf32 if its max stays <~0.85 across seeds; else restrict to n>=1024.")
