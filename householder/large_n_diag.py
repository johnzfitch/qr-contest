"""large_n_diag — confirm the n2048 re-route + answer the 3 diagnostics:
  Q1 device fill: structural (panel kernel launches `batch` CTAs → 8/148 SMs = 5.4%).
  Q2 panel/trailing split of fused-#7 at n2048 b8: panel = full - cuBLAS-trailing.
  Q3 torch.linalg.qr vs torch.geqrf at the large-n shapes (cheap alt check).

  source /workspace/qr/env.sh && python householder/large_n_diag.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check        # noqa: E402


def _imp(p):
    s = importlib.util.spec_from_file_location(p.stem, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


routed = _imp(QRPY / "householder" / "submission_routed.py")
fused = _imp(QRPY / "householder" / "kqr_fused_v2.py")


def _evt(): return torch.cuda.Event(enable_timing=True)


def _time(fn, A, reps=10):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def cublas_trailing_seq(A, nb):
    """Time ONLY the cuBLAS WY trailing sequence for a full n-factorization (random V/T/C)."""
    b, n, _ = A.shape
    steps = []
    p0 = 0
    while p0 < n:
        kb = min(nb, n - p0); pe = p0 + kb
        if pe < n: steps.append((n - p0, kb, n - pe))
        p0 = pe
    tens = []
    for si, (m, k, ncol) in enumerate(steps):
        g = torch.Generator(device="cuda").manual_seed(si)
        V = torch.randn(b, m, k, device="cuda", generator=g)
        T = torch.triu(torch.randn(b, k, k, device="cuda", generator=g))
        C = torch.randn(b, m, ncol, device="cuda", generator=g)
        tens.append((V.contiguous(), T.contiguous(), C.contiguous()))

    def run(_):
        for (V, T, C) in tens:
            W = torch.matmul(V.transpose(1, 2), C); W = torch.matmul(T.transpose(1, 2), W); C.sub_(torch.matmul(V, W))
    return _time(run, None), len(steps)


def run():
    for (b, n, nb) in [(8, 2048, 24), (2, 4096, 8)]:
        print(f"\n===== n={n} b={b} (fused nb={nb}) =====")
        A = make_batch(b, n, 1, "dense", seed=0).cuda().contiguous()

        # routed submission correctness + timing
        H, tau = routed.custom_kernel(A)
        fr, og, ft, ot, ps = check(A, H, tau)
        print(f"  routed.custom_kernel: pass={int(ps.sum())}/{b} margin={max(fr/ft, og/ot):.4f}")
        t_routed = _time(routed.custom_kernel, A)
        t_geqrf = _time(torch.geqrf, A)
        t_full_fused = _time(lambda x: fused.custom_kernel(x, 32), A)
        print(f"  routed         = {t_routed:8.3f} ms")
        print(f"  geqrf          = {t_geqrf:8.3f} ms   (routed/geqrf = {t_routed/t_geqrf:.2f}x)")
        print(f"  fused-#7 direct= {t_full_fused:8.3f} ms")

        # Q3: linalg.qr alt
        try:
            t_lqr = _time(lambda x: torch.linalg.qr(x, mode="reduced"), A)
            print(f"  Q3 linalg.qr   = {t_lqr:8.3f} ms   (vs geqrf {t_geqrf/t_lqr:.2f}x)")
        except Exception as ex:
            print(f"  Q3 linalg.qr   = FAILED ({ex})")

        # Q2: panel/trailing split
        t_trail, nsteps = cublas_trailing_seq(A, nb)
        t_panel = t_full_fused - t_trail
        print(f"  Q2 split: cuBLAS-trailing={t_trail:7.3f} ms ({nsteps} steps) -> panel~={t_panel:7.3f} ms "
              f"({100*t_panel/t_full_fused:.0f}% panel)")
        print(f"  Q1 fill: panel launches {b} CTAs / 148 SMs = {100*b/148:.1f}% during panel phase")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    run()
