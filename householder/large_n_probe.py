"""large_n_probe — does ANYTHING beat torch.geqrf at the batch-starved large-n
board shapes (n2048 b8, n4096 b2)? These two shapes dominate the geomean
(board: 76.8 / 52.0 ms; ln(76.8)=4.34 is the single largest geomean term).

The #7 custom panel is one-CTA-per-matrix → fills only 8/2 SMs of 148 during
panel work (batch-parallelism gone), AND its SMEM panel forces tiny nb at large m.
geqrf (cuSOLVER) parallelizes WITHIN a matrix's panel. Measure the gap and whether
any torch-level restructuring (explicit nb via geqrf-panel + ormqr-trailing) wins.

  source /workspace/qr/env.sh && python householder/large_n_probe.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check        # noqa: E402

# board large-n shapes (batch, n)
SHAPES = [(8, 2048), (2, 4096)]


def _import(modfile):
    spec = importlib.util.spec_from_file_location(modfile.stem, modfile)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


fused = _import(QRPY / "householder" / "kqr_fused_v2.py")


def baseline_geqrf(A):
    return torch.geqrf(A)


def blocked_geqrf_panel(A, nb):
    """Blocked QR: panel via geqrf(narrow block), trailing via ormqr (Q^T C).
    Exposes the nb knob cuSOLVER picks internally. Same (H,tau) layout."""
    H = A.contiguous().clone()
    b, n, _ = H.shape
    tau = torch.zeros(b, n, device=H.device, dtype=H.dtype)
    for p0 in range(0, n, nb):
        kb = min(nb, n - p0); pe = p0 + kb
        panel = H[:, p0:, p0:pe].contiguous()
        Hp, taup = torch.geqrf(panel)
        H[:, p0:, p0:pe] = Hp
        tau[:, p0:pe] = taup
        if pe < n:
            C = H[:, p0:, pe:].contiguous()
            H[:, p0:, pe:] = torch.ormqr(Hp, taup, C, left=True, transpose=True)
    return H, tau


def _evt():
    return torch.cuda.Event(enable_timing=True)


def _time(fn, A, reps=10):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def correctness(A, fn, label):
    H, tau = fn(A)
    fr, og, ft, ot, ps = check(A, H, tau)
    margin = max(fr / ft, og / ot)
    print(f"    {label:28s} pass={int(ps.sum())}/{A.shape[0]}  margin={margin:.4f}")
    return ps.all().item()


def run():
    for (b, n) in SHAPES:
        print(f"\n== n={n} b={b} ==")
        A = make_batch(b, n, 1, "dense", seed=0).cuda().contiguous()

        print("  correctness (vs geqrf reference gate):")
        correctness(A, baseline_geqrf, "geqrf")
        for nb in (64, 128, 256):
            correctness(A, lambda x, nb=nb: blocked_geqrf_panel(x, nb), f"blocked-geqrf-panel nb={nb}")
        correctness(A, lambda x: fused.custom_kernel(x, 32), "fused-#7 nb=32")

        print("  timing:")
        tg = _time(baseline_geqrf, A)
        print(f"    {'geqrf (baseline)':28s} {tg:8.3f} ms   1.00x")
        for nb in (64, 128, 256, 512):
            t = _time(lambda x, nb=nb: blocked_geqrf_panel(x, nb), A)
            print(f"    {'blocked-geqrf-panel nb=' + str(nb):28s} {t:8.3f} ms   {tg/t:.2f}x")
        for nb in (16, 32):
            t = _time(lambda x, nb=nb: fused.custom_kernel(x, nb), A)
            print(f"    {'fused-#7 nb=' + str(nb):28s} {t:8.3f} ms   {tg/t:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    run()
