"""Speed landscape: torch.geqrf baseline vs. CholeskyQR component costs, on the B200.
Tells us whether CholeskyQR+reconstruction can beat geqrf and which stage to hand-roll.
Run:  python dev/bench.py
"""
import torch
from oracle import make_batch, SHAPES

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"


def t_ms(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def comps(A):
    n = A.shape[-1]
    I = torch.eye(n, device=DEV)
    out = {}
    out["geqrf"] = t_ms(lambda: torch.geqrf(A))
    out["gram"] = t_ms(lambda: A.mT @ A)
    M = A.mT @ A
    out["chol"] = t_ms(lambda: torch.linalg.cholesky_ex(M))
    R = torch.linalg.cholesky_ex(M)[0].mT
    out["solve"] = t_ms(lambda: torch.linalg.solve_triangular(R, A, upper=True, left=False))
    Q = torch.linalg.solve_triangular(R, A, upper=True, left=False)
    # cheap reconstruction proxy cost: one geqrf(Q) + one matmul (upper bound; LU kernel will beat it)
    out["recon~geqrf(Q)"] = t_ms(lambda: torch.geqrf(Q))
    out["cqr_core(g+c+s)"] = out["gram"] + out["chol"] + out["solve"]
    return out


print(f"{'shape':24s} {'geqrf':>9s} {'gram':>8s} {'chol':>8s} {'solve':>8s} "
      f"{'g+c+s':>8s} {'rec~gq(Q)':>10s}  speedup(geqrf/core)")
for (b, n, cond, case) in SHAPES:
    if case != "dense":
        continue  # focus on the clean speed picture first
    A = make_batch(b, n, cond, case, seed=0)
    c = comps(A)
    su = c["geqrf"] / c["cqr_core(g+c+s)"]
    print(f"b{b}_n{n}_{case:6s}{'':6s} {c['geqrf']:9.3f} {c['gram']:8.3f} {c['chol']:8.3f} "
          f"{c['solve']:8.3f} {c['cqr_core(g+c+s)']:8.3f} {c['recon~geqrf(Q)']:10.3f}  {su:5.2f}x")
