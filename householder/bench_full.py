"""Validate + benchmark the full submission over all 12 shapes; compare to geqrf, report geomean."""
import math
import torch
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check, SHAPES
from submission_full import custom_kernel


def t_ms(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"{'shape':26s} {'pass':>9s} {'factor':>9s} {'ortho':>9s} {'ours(ms)':>9s} "
      f"{'geqrf(ms)':>10s} {'speedup':>8s}")
ours_log, gq_log = [], []
allpass = True
for (b, n, cond, case) in SHAPES:
    A = make_batch(b, n, cond, case, seed=0)
    H, tau = custom_kernel(A)
    fr, og, ft, ot, ps = check(A, H, tau)
    ok = bool(ps.all()); allpass &= ok
    t_ours = t_ms(lambda: custom_kernel(A))
    t_gq = t_ms(lambda: torch.geqrf(A))
    ours_log.append(math.log(t_ours)); gq_log.append(math.log(t_gq))
    tag = f"b{b}_n{n}_{case}"
    print(f"{tag:26s} {str(ok)+' '+str(int(ps.sum()))+'/'+str(len(ps)):>9s} "
          f"{fr:9.2e} {og:9.2e} {t_ours:9.3f} {t_gq:10.3f} {t_gq/t_ours:7.2f}x")

print(f"\nALL PASS: {allpass}")
print(f"geomean ours : {math.exp(sum(ours_log)/len(ours_log))*1000:.1f} us")
print(f"geomean geqrf: {math.exp(sum(gq_log)/len(gq_log))*1000:.1f} us")
print(f"geomean speedup: {math.exp((sum(gq_log)-sum(ours_log))/len(ours_log)):.2f}x")
