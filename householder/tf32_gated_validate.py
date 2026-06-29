"""Validate tf32_gated_sub: 12-board + 9-stress, margins, geomean. Self-gating tf32."""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check, SHAPES                  # noqa: E402

spec = importlib.util.spec_from_file_location("tg", QRPY / "householder" / "tf32_gated_sub.py")
tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)


def _time(fn, A, reps=30):
    for _ in range(5): fn(A)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(reps): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("== tf32_gated_sub board validation (self-gating tf32) ==")
    g, worst, allpass = 1.0, 0.0, True
    for b, n, cond, case in SHAPES:
        A = make_batch(b, n, cond, case, seed=0).to("cuda").contiguous()
        t = _time(lambda x: tg.custom_kernel(x), A)
        H, tau = tg.custom_kernel(A)
        fr, og, ft, ot, ps = check(A, H, tau)
        m = max(fr / ft, og / ot); g *= t; worst = max(worst, m); allpass = allpass and bool(ps.all())
        print(f"  {str((b,n,case)):22s} {t:8.3f} ms  marg {m:.4f}  {'OK' if ps.all() else 'FAIL'}")
    print(f"  --> 12-shape geomean {g**(1/len(SHAPES)):.3f} ms   worst margin {worst:.4f}   all_pass={allpass}")

    print("\n== 9-stress on fused shapes (n=512 b640, n=1024 b60), seeds 0..3 ==")
    STRESS = ["dense", "mixed", "rankdef", "clustered", "nearrank", "nearcollinear",
              "rowscaled", "banded", "uppertri"]
    sworst, sfail = 0.0, []
    for n, b, cond in [(512, 640, 2), (1024, 60, 2)]:
        for case in STRESS:
            for s in range(4):
                A = make_batch(b, n, cond, case, seed=s).to("cuda").contiguous()
                H, tau = tg.custom_kernel(A)
                fr, og, ft, ot, ps = check(A, H, tau)
                m = max(fr / ft, og / ot); sworst = max(sworst, m)
                if not ps.all(): sfail.append(f"n{n}/{case}/s{s}")
    print(f"  stress worst margin {sworst:.4f}   fails={sfail if sfail else 'NONE'}")
