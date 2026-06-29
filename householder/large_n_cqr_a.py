"""Phase A — full CQR2 pipeline at the large-n board shapes. A0 showed the CQR2 core
(Gram+chol+trsm) = 26ms (n2048) / 43ms (n4096); the open question is the modLU
reconstruction cost. Measure: correctness + MARGIN (not just pass/fail), full pipeline
timing vs fused-#7 and geqrf, and the robust_cqr / blocked_modlu decomposition.

  source /workspace/qr/env.sh && python householder/large_n_cqr_a.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check        # noqa: E402


def _imp(p):
    s = importlib.util.spec_from_file_location(p.stem, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


kb = _imp(QRPY / "kan_metric" / "kqr_blocked.py")     # pipeline, robust_cqr, blocked_modlu
fused = _imp(QRPY / "householder" / "kqr_fused_v2.py")


def _evt():
    return torch.cuda.Event(enable_timing=True)


def _time(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def run(b, n, fused_pod_ref):
    print(f"\n===== n={n} b={b} =====")
    A = make_batch(b, n, 1, "dense", seed=0).cuda().contiguous()

    # correctness + margin
    H, tau = kb.pipeline(A)
    fr, og, ft, ot, ps = check(A, H, tau)
    margin = max(fr / ft, og / ot)
    frag = "  <-- FRAGILE (margin>0.5)" if margin > 0.5 else ""
    print(f"  CQR2 pipeline correctness: pass={int(ps.sum())}/{b}  MARGIN={margin:.4f}{frag}")

    # timing: full pipeline vs fused-#7 vs geqrf
    t_pipe = _time(lambda: kb.pipeline(A))
    t_fused = _time(lambda: fused.custom_kernel(A, 32))
    t_geqrf = _time(lambda: torch.geqrf(A))
    print(f"  CQR2 pipeline = {t_pipe:7.3f} ms")
    print(f"  fused-#7      = {t_fused:7.3f} ms   (pod ref {fused_pod_ref})")
    print(f"  geqrf         = {t_geqrf:7.3f} ms")
    print(f"  -> CQR2 vs fused = {t_fused/t_pipe:.2f}x   CQR2 vs geqrf = {t_geqrf/t_pipe:.2f}x")

    # decompose: robust_cqr (the GEMM core) vs blocked_modlu (the reconstruction)
    Q, R_chol, ok = kb.robust_cqr(A, passes=2)
    t_cqr = _time(lambda: kb.robust_cqr(A, passes=2))
    t_modlu = _time(lambda: kb.blocked_modlu(Q))
    print(f"  decompose: robust_cqr={t_cqr:7.3f} ms  blocked_modlu={t_modlu:7.3f} ms"
          f"  (modLU = {100*t_modlu/t_pipe:.0f}% of pipeline)")
    return t_pipe, margin


if __name__ == "__main__":
    assert torch.cuda.is_available()
    run(8, 2048, "46.4ms")
    run(2, 4096, "180ms(loses)")
