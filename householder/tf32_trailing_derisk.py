"""tf32-trailing de-risk -- the live sub-lever the wmma probe surfaced.

mma_apply_derisk showed: in-kernel wmma LOSES to cuBLAS at full-apply size, BUT
cuBLAS-tf32 (73us, 5.1x) beats cuBLAS-fp32 (93us, 4.0x) on the trailing apply.
The shipped fused kernel runs the between-panel at::matmul trailing in FP32
(allow_tf32=False). tf32 affects ONLY those trailing GEMMs (the within-panel
geqr2 is our fp32 warp-FMA kernel), so flipping the global flag isolates exactly
the trailing-update contribution -- and tests whether tf32's ~10-bit mantissa
survives the gate on the rank-deficient / clustered scored shapes (the known risk).

Reports, per n512 shape: end-to-end ms (fp32 vs tf32 trailing) AND oracle margin.
Win only if tf32 is materially faster AND margin<1 on ALL 4 (incl rankdef,clustered).

  source /workspace/qr/env.sh && python householder/tf32_trailing_derisk.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check                          # noqa: E402

v3 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v3", QRPY / "householder" / "kqr_fused_v3.py"))
importlib.util.spec_from_file_location("v3", QRPY / "householder" / "kqr_fused_v3.py").loader.exec_module(v3)


def _time(fn, A, reps=30):
    for _ in range(5): fn(A)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(reps): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def margin(A, H, tau):
    fr, og, ft, ot, ps = check(A, H, tau)
    return max(fr / ft, og / ot), bool(ps.all())


if __name__ == "__main__":
    assert torch.cuda.is_available()
    SHAPES = ["dense", "mixed", "rankdef", "clustered"]
    print("== tf32-trailing de-risk  (n=512, b=640)  end-to-end ms + oracle margin ==")
    print(f"{'shape':12s} {'fp32 ms':>9s} {'tf32 ms':>9s} {'speedup':>8s} {'fp32 marg':>10s} {'tf32 marg':>10s} {'tf32 pass':>9s}")
    geo_fp32, geo_tf32 = 1.0, 1.0
    for case in SHAPES:
        A = make_batch(640, 512, 2, case, seed=0).to("cuda").contiguous()

        torch.backends.cuda.matmul.allow_tf32 = False
        t_fp = _time(lambda x: v3.custom_kernel(x, 32), A)
        H, tau = v3.custom_kernel(A, 32); m_fp, p_fp = margin(A, H, tau)

        torch.backends.cuda.matmul.allow_tf32 = True
        t_tf = _time(lambda x: v3.custom_kernel(x, 32), A)
        H, tau = v3.custom_kernel(A, 32); m_tf, p_tf = margin(A, H, tau)
        torch.backends.cuda.matmul.allow_tf32 = False

        geo_fp32 *= t_fp; geo_tf32 *= t_tf
        print(f"{case:12s} {t_fp:9.3f} {t_tf:9.3f} {t_fp/t_tf:7.2f}x {m_fp:10.4f} {m_tf:10.4f} {str(p_tf):>9s}")
    print(f"\n  n512 geomean (4 shapes):  fp32 {geo_fp32**0.25:.3f} ms   tf32 {geo_tf32**0.25:.3f} ms"
          f"   ({(geo_fp32/geo_tf32)**0.25:.2f}x)")
    print("  SHIP only if tf32 faster AND every tf32 margin < 1.0 (incl rankdef, clustered).")
