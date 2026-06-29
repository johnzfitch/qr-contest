"""The continued-fraction / Lanczos question, reduced to its one empirical claim:
a 3-term recurrence has NO trailing update -> does it fill the batch (win) or
serialize into n latency-bound launches (lose)? Measure the irreducible cost.

  source /workspace/qr/env.sh && python householder/lanczos_latency_probe.py
"""
import sys, pathlib, time
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
torch.backends.cuda.matmul.allow_tf32 = False


def t_ms(fn, reps=1):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / reps * 1e3


def run(b, n):
    print(f"\n===== n={n} b={b}  (fused-#7 QR = ~46 ms; geqrf = ~76 ms) =====")
    A = torch.randn(b, n, n, device="cuda")
    M = (A.mT @ A).contiguous()
    v0 = torch.randn(b, n, 1, device="cuda")

    # (1) THE RECURRENCE: n sequential batched matvecs + normalize (irreducible
    #     serial chain of ANY 3-term Lanczos/bidiag method -- each step needs the prior).
    def lanczos_chain():
        v = v0.clone(); vp = torch.zeros_like(v)
        for _ in range(n):
            w = M @ v                              # the matvec the "no trailing update" pays
            w = w - (v * w).sum(1, keepdim=True) * v
            w = w / (w.norm(dim=1, keepdim=True) + 1e-30)
            vp, v = v, w
        return v
    t_rec = t_ms(lanczos_chain)

    # (2) ONE device-filling GEMM (trailing-update-shaped): what we pay INSTEAD.
    t_gemm = t_ms(lambda: M @ A, reps=10)

    # (3) Golub-Kahan would be 2n matvecs (A@v and A^T@u): the bidiag variant.
    print(f"  (1) 3-term recurrence: {n} serial batched matvecs+normalize = {t_rec:8.1f} ms")
    print(f"  (2) one big batched GEMM (M@A, device-filling)              = {t_gemm:8.3f} ms")
    print(f"  -> recurrence / one-GEMM = {t_rec/t_gemm:6.0f}x ;  recurrence / fused-46ms = {t_rec/46:.1f}x")
    print(f"     (Golub-Kahan bidiag = 2x the recurrence = ~{2*t_rec:.0f} ms)")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    run(8, 2048)
