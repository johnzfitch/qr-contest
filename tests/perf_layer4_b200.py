"""Layer 4 — performance (B200 ONLY). Ranks speed AFTER accuracy is settled on CPU.

  L4.1  geomean ms over the 12 ranked shapes per PASSING arch -> leaderboard
        (uses oracle generators so the timed inputs == the gated inputs).
  L4.2  the load-bearing micro-measurements the design chats flagged:
          - geqrf(Q) vs geqrf(A) at b640/n512  (the CQR-recon premise)
          - bf16-trailing vs fp32-trailing speedup per shape (the actual lever)
          - panel-time fraction at n=512 (is the recursive E-G panel worth building?)
          - fp8 vs bf16 geomean (caveat: fp8 sims through fp32 until _scaled_mm is wired)
  L4.3  every contender must beat the fp32 HH control (arch 1) on geomean.

Requires CUDA. Run on the pod:
  /opt/conda/bin/python perf_layer4_b200.py            # full ARCH_PASS
  /opt/conda/bin/python perf_layer4_b200.py 1 4 10     # specific archs
  /opt/conda/bin/python perf_layer4_b200.py --quick    # dense shapes only, archs 1,4,10
"""
import math
import sys
import time

import torch

from common import oracle, port, set_arch, arch_label, ARCH_PASS, DEV


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _timed_blocked_hh(A):
    """Mirror of port.blocked_hh with CUDA-event timing split into panel vs trailing.
       Returns (panel_ms, trailing_ms, total_ms). Reads the same globals as the artifact."""
    ev = lambda: torch.cuda.Event(enable_timing=True)
    panel_ms = 0.0
    trail_ms = 0.0
    B, m, n = A.shape
    work = A.double()
    H = work.clone()
    tau = torch.zeros(B, n, dtype=work.dtype, device=work.device)
    t_all0, t_all1 = ev(), ev()
    t_all0.record()
    for k in range(0, n, port.NB):
        kb = min(port.NB, n - k)
        p0, p1 = ev(), ev()
        p0.record()
        panel = H[:, k:, k:k + kb].clone()
        Vp, taup = torch.geqrf(panel)
        p1.record()
        H[:, k:, k:k + kb] = Vp
        tau[:, k:k + kb] = taup
        if k + kb < n:
            V = torch.tril(Vp, -1)
            ar = torch.arange(kb, device=V.device)
            V[:, ar, ar] = 1.0
            T = port._build_T(V, taup)
            trailing = H[:, k:, k + kb:]
            q0, q1 = ev(), ev()
            q0.record()
            if port._is_edge(k, kb, n):
                W = V.transpose(-2, -1) @ trailing
                TW = T.transpose(-2, -1) @ W
                H[:, k:, k + kb:] = trailing - V @ TW
            else:
                W = port.qmm(V.transpose(-2, -1), trailing)
                TW = T.transpose(-2, -1) @ W
                H[:, k:, k + kb:] = trailing - port.qmm(V, TW)
            q1.record()
            torch.cuda.synchronize()
            trail_ms += q0.elapsed_time(q1)
        torch.cuda.synchronize()
        panel_ms += p0.elapsed_time(p1)
    t_all1.record()
    torch.cuda.synchronize()
    return panel_ms, trail_ms, t_all0.elapsed_time(t_all1)


def main():
    if DEV != "cuda":
        print("Layer 4 requires CUDA (B200). Skipping on CPU.")
        sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    archs = [int(a) for a in args] or ([1, 4, 10] if quick else ARCH_PASS)
    shapes = [s for s in oracle.SHAPES if s[3] == "dense"] if quick else oracle.SHAPES

    print(f"=== Layer 4 — perf (B200: {torch.cuda.get_device_name(0)}) ===")
    print(f"  archs={archs}  shapes={len(shapes)}  quick={quick}\n")

    # ---- L4.1 leaderboard (only time archs that pass the gate on every shape) -------
    rows = []
    for arch in archs:
        set_arch(arch)
        per_shape = []
        passed_all = True
        for (b, n, cond, case) in shapes:
            A = oracle.make_batch(b, n, cond, case, seed=0)
            H, tau = port.custom_kernel(A)
            _, _, _, _, ps = oracle.check(A, H, tau)
            if not ps.all().item():
                passed_all = False
            per_shape.append(bench(lambda: port.custom_kernel(A)))
        gm = geomean(per_shape)
        rows.append((arch, gm, passed_all, per_shape))
    rows.sort(key=lambda r: r[1])

    print("  L4.1 geomean leaderboard (ms; lower is better)")
    print(f"  {'arch':26s} {'geomean':>9s}  gate  per-shape ms")
    ctrl_gm = next((gm for (a, gm, _, _) in rows if a == 1), None)
    for (arch, gm, ok, per) in rows:
        beat = "" if ctrl_gm is None or arch == 1 else (f"  {ctrl_gm / gm:.2f}x vs a1" if gm < ctrl_gm else f"  SLOWER than a1")
        pers = " ".join(f"{m:.2f}" for m in per)
        print(f"  {arch_label(arch):26s} {gm:9.3f}  {'OK ' if ok else 'FAIL'}  [{pers}]{beat}")

    # ---- L4.3 contenders must beat the fp32 control -------------------------------
    if ctrl_gm is not None:
        print("\n  L4.3 beat-the-fp32-control check")
        for (arch, gm, ok, _) in rows:
            if arch == 1:
                continue
            verdict = "PASS" if gm < ctrl_gm else "POINTLESS (>= control)"
            print(f"    [{verdict}] {arch_label(arch)}  {ctrl_gm / gm:.2f}x")

    # ---- L4.2 micro-measurements --------------------------------------------------
    print("\n  L4.2 micro-measurements")

    # (a) geqrf(Q) vs geqrf(A) at b640/n512 — the CQR-recon premise
    A = oracle.make_batch(640, 512, 2, "dense", seed=0)
    Q, _R, _ = oracle.cholesky_qr(A, passes=2)
    t_gA = bench(lambda: torch.geqrf(A))
    t_gQ = bench(lambda: torch.geqrf(Q))
    print(f"    geqrf(A) b640/n512 = {t_gA:.3f} ms ;  geqrf(Q) = {t_gQ:.3f} ms  "
          f"(recon overhead {t_gQ / t_gA:.2f}x)")

    # (b) bf16-trailing vs fp32-trailing per shape (the lever) — arch4 vs arch1
    print("    bf16(a4) vs fp32(a1) trailing speedup per shape:")
    for (b, n, cond, case) in shapes:
        A = oracle.make_batch(b, n, cond, case, seed=0)
        set_arch(1)
        t1 = bench(lambda: port.custom_kernel(A))
        set_arch(4)
        t4 = bench(lambda: port.custom_kernel(A))
        print(f"      b{b}_n{n}_{case:9s} fp32={t1:7.2f} bf16={t4:7.2f}  {t1 / t4:.2f}x")

    # (c) panel-time fraction at n=512 (recursive E-G panel worth building?)
    set_arch(4)
    A = oracle.make_batch(640, 512, 2, "dense", seed=0)
    pm, tm, total = _timed_blocked_hh(A)
    print(f"    panel-fraction n=512 (a4): panel={pm:.2f}ms trailing={tm:.2f}ms "
          f"-> panel {100 * pm / (pm + tm):.0f}% of (panel+trailing)")

    # (d) fp8 vs bf16 geomean (caveat) — arch10 vs arch4
    fp8 = next((gm for (a, gm, _, _) in rows if a == 10), None)
    bf16 = next((gm for (a, gm, _, _) in rows if a == 4), None)
    if fp8 and bf16:
        print(f"    fp8(a10) vs bf16(a4) geomean: {bf16 / fp8:.2f}x  "
              f"(NOTE: fp8 sims through fp32 until torch._scaled_mm/CuTe is wired -> expect ~1x here)")

    print("\nLayer 4 complete (advisory — informs which arch to ship).")


if __name__ == "__main__":
    main()
