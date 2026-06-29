"""Full 12-board tf32-trailing decision probe: geomean win + worst margin.

Runs the SHIPPED routing (jerry_owes_me_lunch.custom_kernel) over all 12 board
shapes, fp32 trailing vs tf32 trailing, reporting per-shape ms + oracle margin and
the 12-shape geomean. The decision number: does tf32 move the geomean enough to
justify the margin it spends, and does EVERY shape stay under the gate (<1.0)?

  source /workspace/qr/env.sh && python householder/tf32_board_derisk.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check, SHAPES                  # noqa: E402

spec = importlib.util.spec_from_file_location("jerry", QRPY / "householder" / "jerry_owes_me_lunch.py")
jerry = importlib.util.module_from_spec(spec); spec.loader.exec_module(jerry)


def _time(fn, A, reps=30):
    for _ in range(5): fn(A)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(reps): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def run(case_tuple, tf32):
    b, n, cond, case = case_tuple
    A = make_batch(b, n, cond, case, seed=0).to("cuda").contiguous()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    t = _time(lambda x: jerry.custom_kernel(x), A)
    H, tau = jerry.custom_kernel(A)
    fr, og, ft, ot, ps = check(A, H, tau)
    torch.backends.cuda.matmul.allow_tf32 = False
    return t, max(fr / ft, og / ot), bool(ps.all())


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("== full 12-board tf32-trailing decision  (shipped routing) ==")
    print(f"{'shape':18s} {'fp32 ms':>9s} {'tf32 ms':>9s} {'spd':>6s} {'tf32 marg':>10s} {'pass':>6s}")
    gf, gt, worst_tf = 1.0, 1.0, 0.0
    for st in SHAPES:
        tf, mf, pf = run(st, False)
        tt, mt, pt = run(st, True)
        gf *= tf; gt *= tt; worst_tf = max(worst_tf, mt)
        tag = "" if pt else "  <-- FAIL"
        print(f"{str((st[0],st[1],st[3])):18s} {tf:9.3f} {tt:9.3f} {tf/tt:5.2f}x {mt:10.4f} {str(pt):>6s}{tag}")
    n = len(SHAPES)
    print(f"\n  12-shape geomean:  fp32 {gf**(1/n)*1e0:.3f} ms   tf32 {gt**(1/n):.3f} ms   ({(gf/gt)**(1/n):.2f}x)")
    print(f"  worst tf32 margin across board = {worst_tf:.4f}   (gate = 1.0; want comfortable headroom)")
