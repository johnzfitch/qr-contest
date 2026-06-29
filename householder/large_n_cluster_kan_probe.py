"""Timeboxed probe: can kqr_cluster_kan run at n=2048 b8? It holds 2*n*w = 2n^2/C floats
of SMEM per CTA. Designed for n<=512. Check whether any allowed C (<=16) fits B200 SMEM,
and confirm empirically (n=512 sanity + n=2048 attempts).

  source /workspace/qr/env.sh && python householder/large_n_cluster_kan_probe.py
"""
import sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check        # noqa: E402


def _imp(p):
    s = importlib.util.spec_from_file_location(p.stem, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


kk = _imp(QRPY / "kan_metric" / "kqr_cluster_kan.py")
ext = kk._build(4, 3)        # compile full pipeline kernel


def smem_mb(n, C):
    return 2 * n * (n // C) * 4 / 1e6


print(f"B200 dynamic-SMEM ceiling ~0.228 MB/CTA. smem = 2*n*(n/C) floats per CTA.")
for (b, n, C) in [(8, 512, 16), (8, 2048, 16), (8, 2048, 8), (8, 2048, 4)]:
    if n % C or (n // C) % 16:
        print(f"  n={n} C={C}: invalid (n%C or w%16)"); continue
    A = make_batch(b, n, 1, "dense", seed=0).cuda().contiguous()
    try:
        H, tau = ext.cluster_qr(A, C)
        fr, og, ft, ot, ps = check(A, H, tau)
        print(f"  n={n} C={C} w={n//C}: smem={smem_mb(n,C):.2f}MB  RUNS  pass={int(ps.sum())}/{b} margin={max(fr/ft,og/ot):.4f}")
    except Exception as e:
        print(f"  n={n} C={C} w={n//C}: smem={smem_mb(n,C):.2f}MB  FAIL: {str(e)[:90]}")
