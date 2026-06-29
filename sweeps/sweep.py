#!/usr/bin/env python3
"""
sweep.py  —  run all 20 architectures in qr_hh_portfolio.py across the contest
benchmark shapes on a B200, verify each against the geqrf gate, and rank the
PASSING ones by geometric-mean runtime (the contest metric).

Usage on B200:
    python sweep.py                 # all archs, all shapes
    python sweep.py --arch 2 7 11   # subset
    python sweep.py --quick         # dense shapes only, 1 rep

Reads qr_hh_portfolio.py by re-importing with QR_ARCH set per run (subprocess, so
the _HAS_CUTE / arch globals re-bind cleanly each time).
"""
import argparse, json, math, os, subprocess, sys, time

BENCH_SHAPES = [
    dict(batch=20,  cond=1, n=32),
    dict(batch=40,  cond=1, n=176),
    dict(batch=40,  cond=1, n=352),
    dict(batch=640, cond=2, n=512),
    dict(batch=60,  cond=2, n=1024),
    dict(batch=8,   cond=1, n=2048),
    dict(batch=2,   cond=1, n=4096),
    dict(batch=640, cond=2, n=512, case="mixed"),
    dict(batch=60,  cond=2, n=1024, case="mixed"),
    dict(batch=640, cond=0, n=512, case="rankdef"),
    dict(batch=640, cond=0, n=512, case="clustered"),
    dict(batch=60,  cond=0, n=1024, case="nearrank"),
]

RUNNER = r'''
import os, sys, json, time, math
import torch
os.environ["QR_ARCH"] = sys.argv[1]
import importlib.util
spec = importlib.util.spec_from_file_location("qp", sys.argv[2])
qp = importlib.util.module_from_spec(spec); spec.loader.exec_module(qp)

def make_A(shape):
    B, n, cond = shape["batch"], shape["n"], shape.get("cond", 1)
    case = shape.get("case", "dense")
    g = torch.Generator(device="cuda").manual_seed(1234)
    A = torch.randn(B, n, n, device="cuda", generator=g)
    if case == "dense" or case == "mixed":
        scale = torch.logspace(0, -cond, n, device="cuda")
        A = A * scale.view(1, 1, n)
    if case == "rankdef":
        A[:, :, n//2:] = A[:, :, :n//2] @ torch.randn(B, n//2, n-n//2, device="cuda", generator=g) * 1e-3
    if case == "clustered":
        scale = torch.cat([torch.full((n//2,), 1.0), torch.full((n-n//2,), 1e-6)]).to("cuda")
        A = A * scale.view(1,1,n)
    if case == "nearrank":
        A[:, :, -1] = A[:, :, 0] + 1e-7 * torch.randn(B, n, device="cuda", generator=g)
    return A.contiguous()

def gate(A, H, tau):
    n = A.shape[-1]
    eps = 1.1920929e-07
    Ad, Hd = A.double(), H.double()
    Q = torch.linalg.householder_product(Hd, tau.double())
    R = torch.triu(Hd)
    fr = (R - Q.transpose(-2,-1) @ Ad).abs()
    Anorm = Ad.abs().sum(-2, keepdim=True).clamp_min(1e-300)
    res_ok = (fr.sum(-2) <= 20*n*eps * Anorm.squeeze(-2)).all().item()
    orth = (Q.transpose(-2,-1) @ Q - torch.eye(n, device="cuda", dtype=torch.float64)).abs()
    orth_ok = (orth.sum(-2) <= 100*n*eps * 1.0).all().item()
    finite = torch.isfinite(H).all().item() and torch.isfinite(tau).all().item()
    return bool(res_ok and orth_ok and finite)

shape = json.loads(sys.argv[3])
A = make_A(shape)
# warmup
try:
    H, tau = qp.custom_kernel(A.clone())
    torch.cuda.synchronize()
    ok = gate(A, H, tau)
except Exception as e:
    print(json.dumps(dict(ok=False, err=str(e)[:200], ms=float("inf")))); sys.exit(0)

reps = 5
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(reps):
    H, tau = qp.custom_kernel(A.clone())
torch.cuda.synchronize(); t1 = time.perf_counter()
ms = (t1 - t0) / reps * 1e3
print(json.dumps(dict(ok=ok, ms=ms, err="")))
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=int, nargs="*", default=list(range(1, 21)))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--portfolio", default=os.path.join(os.path.dirname(__file__), "qr_hh_portfolio.py"))
    args = ap.parse_args()

    shapes = BENCH_SHAPES
    if args.quick:
        shapes = [s for s in BENCH_SHAPES if s.get("case", "dense") == "dense"]

    # write runner to a temp file
    runner_path = os.path.join(os.path.dirname(args.portfolio) or ".", "_runner.py")
    with open(runner_path, "w") as f:
        f.write(RUNNER)

    results = {}
    for arch in args.arch:
        per_shape = []
        all_pass = True
        for shp in shapes:
            out = subprocess.run(
                [sys.executable, runner_path, str(arch), args.portfolio, json.dumps(shp)],
                capture_output=True, text=True, timeout=600,
            )
            line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "{}"
            try:
                r = json.loads(line)
            except Exception:
                r = dict(ok=False, ms=float("inf"), err=out.stderr[-200:])
            per_shape.append((shp, r))
            if not r.get("ok", False):
                all_pass = False
        # geomean over passing shapes (contest metric)
        mss = [r["ms"] for _, r in per_shape if r.get("ok") and math.isfinite(r["ms"])]
        geo = math.exp(sum(math.log(m) for m in mss) / len(mss)) if mss else float("inf")
        results[arch] = dict(all_pass=all_pass, geomean_ms=geo, detail=per_shape)
        tag = "PASS" if all_pass else "FAIL"
        print(f"ARCH {arch:2d}  [{tag}]  geomean={geo:8.3f} ms  "
              f"({len(mss)}/{len(shapes)} shapes ok)")

    # ranked leaderboard of fully-passing archs
    print("\n=== RANKED (fully-passing only) ===")
    ranked = sorted(
        [(a, r["geomean_ms"]) for a, r in results.items() if r["all_pass"]],
        key=lambda x: x[1],
    )
    for i, (a, g) in enumerate(ranked, 1):
        print(f"  {i:2d}. ARCH {a:2d}   {g:8.3f} ms")
    if not ranked:
        print("  (none fully passed — inspect per-shape detail below)")

    # dump full detail
    with open("sweep_results.json", "w") as f:
        json.dump({str(k): {"all_pass": v["all_pass"], "geomean_ms": v["geomean_ms"],
                            "detail": [(s, r) for s, r in v["detail"]]}
                   for k, v in results.items()}, f, indent=2, default=str)
    print("\nFull detail -> sweep_results.json")

if __name__ == "__main__":
    main()
