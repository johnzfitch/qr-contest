"""Phase A0 — the cheapest kill. Does cuSOLVER serialize batched cholesky/trsm at small
batch (like geqrf)? If chol+trsm alone approach the 52ms budget at b8 n2048 / b2 n4096,
CQR2 cannot win and we stop. Pure torch, no pipeline.

Serialization test: compare b=8 vs b=1 per-matrix time. Linear scaling (t_b8 ≈ 8·t_b1)
⇒ serialized ⇒ no win. Flat (t_b8 ≈ t_b1) ⇒ batch-parallel ⇒ room.

  source /workspace/qr/env.sh && python householder/large_n_cqr_a0.py
"""
import sys, pathlib
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch        # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def _evt():
    return torch.cuda.Event(enable_timing=True)


def _time(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def run_shape(b, n, budget):
    print(f"\n===== n={n} b={b}  (current shipped route = {budget:.1f} ms) =====")
    A = make_batch(b, n, 1, "dense", seed=0).cuda().contiguous()
    A1 = A[:1].contiguous()                      # single-matrix for serialization test

    gram_b = _time(lambda: A.mT @ A)
    M = (A.mT @ A).contiguous()
    M1 = (A1.mT @ A1).contiguous()

    chol_b = _time(lambda: torch.linalg.cholesky_ex(M))
    chol_1 = _time(lambda: torch.linalg.cholesky_ex(M1))
    L = torch.linalg.cholesky_ex(M)[0]
    L1 = torch.linalg.cholesky_ex(M1)[0]

    trsm_b = _time(lambda: torch.linalg.solve_triangular(L, A.mT, upper=False, left=True))
    trsm_1 = _time(lambda: torch.linalg.solve_triangular(L1, A1.mT, upper=False, left=True))

    def ser(tb, t1):                              # serialization factor: 1.0 = perfect parallel, b = fully serial
        return (tb / t1) if t1 > 0 else float("nan")

    print(f"  Gram  A^T A          : {gram_b:7.3f} ms")
    print(f"  cholesky_ex          : {chol_b:7.3f} ms   (b1={chol_1:6.3f}; serial-factor {ser(chol_b, chol_1):.1f}/{b})")
    print(f"  solve_triangular     : {trsm_b:7.3f} ms   (b1={trsm_1:6.3f}; serial-factor {ser(trsm_b, trsm_1):.1f}/{b})")
    cqr2 = 2 * (gram_b + chol_b + trsm_b)
    print(f"  -> CQR2 core (2x(Gram+chol+trsm)) = {cqr2:7.3f} ms   vs budget {budget:.1f} ms"
          f"   [{'ROOM for modLU' if cqr2 < budget else 'ALREADY OVER -> KILL'}]")
    return cqr2


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("Serial-factor near 1.0 = batch-parallel (good); near b = serialized like geqrf (bad).")
    run_shape(8, 2048, 52.5)
    run_shape(2, 4096, 52.2)
