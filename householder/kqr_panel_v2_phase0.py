"""kqr_panel_v2 — PHASE 0: measured baseline split (replaces the inferred 27.2/1.3).

Isolates, with CUDA events, for the existing scalar panel_geqrt artifact:
  panel-only   : loop that calls ONLY the panel kernel across all p0 (no trailing).
                 Valid for timing — the kernel does the same m*nb*nb work regardless
                 of whether the trailing update ran (only the values differ).
  full QR      : custom_kernel (matches the board number).
  trailing+orch: full - panel-only.
Plus static resources from a -Xptxas=-v recompile (registers/thread, SMEM/CTA,
spills) and active-CTAs/SM from the occupancy API. ncu metrics (bank conflicts,
barrier stalls, achieved FP32 throughput) are PERMS-BLOCKED on this pod -> N/A.

  source /workspace/qr/env.sh && python householder/kqr_panel_v2_phase0.py
"""
import sys, pathlib, importlib.util, time
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch                                  # noqa: E402

spec = importlib.util.spec_from_file_location("art", QRPY / "householder" / "submission_full.py")
art = importlib.util.module_from_spec(spec); spec.loader.exec_module(art)
K, _nb_for = art._K, art._nb_for

SHAPES = [(176, 40), (352, 40), (512, 640), (1024, 60)]


def _evt():
    return torch.cuda.Event(enable_timing=True)


def panel_only_ms(A, reps=20):
    b, n, _ = A.shape
    B = A.contiguous().clone()
    tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    def one():
        p0 = 0
        while p0 < n:
            nb = min(_nb_for(n), n - p0)
            T = torch.empty(b, nb, nb, device=A.device, dtype=A.dtype)
            K.panel_geqrt(B, T, tau, p0, nb)
            p0 += nb
    for _ in range(3): one()
    torch.cuda.synchronize()
    s, e = _evt(), _evt(); s.record()
    for _ in range(reps): one()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def full_ms(A, reps=20):
    for _ in range(3): art.custom_kernel(A)
    torch.cuda.synchronize()
    s, e = _evt(), _evt(); s.record()
    for _ in range(reps): art.custom_kernel(A)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def n_panels(n):
    nb = _nb_for(n); return -(-n // nb), nb


def main():
    assert torch.cuda.is_available()
    print(f"{'shape':14s} {'nb':>3s} {'#pan':>4s} | {'panel-only':>10s} {'full QR':>9s} "
          f"{'trail+orch':>10s} {'panel%':>7s}")
    print("-" * 72)
    for n, b in SHAPES:
        A = make_batch(b, n, 2, "dense", seed=0).to("cuda").contiguous()
        p = panel_only_ms(A); f = full_ms(A)
        npan, nb = n_panels(n)
        rest = max(0.0, f - p)
        print(f"n={n:<5d}b={b:<5d} {nb:>3d} {npan:>4d} | {p:9.3f}m {f:8.3f}m "
              f"{rest:9.3f}m {100*p/f:6.1f}%")

    # ---- static resources via -Xptxas=-v recompile (registers/smem/spills) ----
    print("\n== static panel-kernel resources (ptxas -v; name forces recompile) ==")
    from torch.utils.cpp_extension import load_inline
    _ = load_inline(name="panel_v2_ptxas", cpp_sources=[art.CPP_SRC], cuda_sources=[art.CUDA_SRC],
                    functions=["panel_geqrt"], extra_cuda_cflags=["-O3", "-Xptxas=-v"], verbose=True)
    print("  (grep the build output above for: 'Used N registers', 'smem', 'spill')")


if __name__ == "__main__":
    main()
