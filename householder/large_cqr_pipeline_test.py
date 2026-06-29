"""Does the existing CQR pipeline (CQR2 -> modLU recon -> H,tau) with tf32 beat
geqrf at the LARGE low-batch shapes (n2048 b8 = 28.8ms, n4096 b2 = 52ms)?

These board shapes are cond-1 (well-conditioned) -> tf32 CQR should pass the gate.
End-to-end incl reconstruction. If it beats geqrf + passes, it's a large-shape win.

  source /workspace/qr/env.sh && python householder/large_cqr_pipeline_test.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common"), str(QRPY / "kan_metric")]
from oracle import make_batch, check                          # noqa: E402

spec = importlib.util.spec_from_file_location("kqrb", QRPY / "kan_metric" / "kqr_blocked.py")
kqrb = importlib.util.module_from_spec(spec); spec.loader.exec_module(kqrb)


def _t(fn, r=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / r


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("== CQR pipeline (tf32) vs geqrf at large board shapes ==")
    print(f"{'shape':18s} {'tf32?':6s} {'pipe ms':>9s} {'geqrf ms':>9s} {'speedup':>8s} {'margin':>8s} {'pass':>6s} {'gate->geqrf':>11s}")
    for b, n, cond, case in [(8, 2048, 1, "dense"), (2, 4096, 1, "dense"), (8, 2048, 2, "mixed")]:
        A = make_batch(b, n, cond, case, seed=0).cuda().contiguous()
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            H, tau = kqrb.pipeline(A)
            fr, og, ft, ot, ps = check(A, H, tau)
            margin = max(fr / ft, og / ot)
            # how many matrices fell back to geqrf inside the pipeline?
            Q, Rc, ok = kqrb.robust_cqr(A)
            I = torch.eye(n, device=A.device).expand_as(Q)
            defect = torch.linalg.matrix_norm((Q.mT @ Q - I).float(), ord=1, dim=(-2, -1))
            good = ok & torch.isfinite(defect) & (defect < 4.0 * 100 * n * kqrb.EPS32)
            nfb = int((~good).sum())
            t_pipe = _t(lambda: kqrb.pipeline(A))
            torch.backends.cuda.matmul.allow_tf32 = False
            t_geqrf = _t(lambda: torch.geqrf(A))
            print(f"{str((b,n,case)):18s} {str(tf32):6s} {t_pipe*1e3:9.2f} {t_geqrf*1e3:9.2f} {t_geqrf/t_pipe:7.2f}x "
                  f"{margin:8.4f} {str(bool(ps.all())):>6s} {nfb:>5d}/{b}")
