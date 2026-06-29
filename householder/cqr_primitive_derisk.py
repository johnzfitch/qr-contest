"""CQR load-bearing-primitive de-risk: can CholeskyQR be fast at b640 n512?

The leader is ~1.2ms (we are 6.9ms). Only tensor-core GEMM-dominated QR gets there;
the family is CholeskyQR (A^T A -> chol -> Q=A R^-1). Our prior CQR2 = 107ms, but that
leaned on torch batched cholesky which may SERIALIZE at b640 like geqrf. This probe
measures the primitives IN ISOLATION at the heavily-weighted n512 b640 shape:
  1. A^T A  fp32 (CUDA core) vs bf16/tf32 (tensor core)   -- the Gram GEMM
  2. cholesky_ex(640,512,512)  -- does it SERIALIZE? compare vs the 1-matrix cost x640
  3. triangular solve  Q = A R^-1
  4. sum ~= one CQR iter; x2 + recon ~= CQR2 cost. Compare to HH 12ms and leader ~1ms.
Also the bf16 A^T A accuracy (cond^2 amplification) note: is fp32-accumulate enough?

  source /workspace/qr/env.sh && python householder/cqr_primitive_derisk.py
"""
import sys, pathlib
import torch

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch                                  # noqa: E402


def _t(fn, r=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / r


if __name__ == "__main__":
    assert torch.cuda.is_available()
    for (B, N) in [(640, 512), (60, 1024), (8, 2048)]:
        A = make_batch(B, N, 2, "dense", seed=0).cuda().contiguous()
        Abf = A.to(torch.bfloat16)
        I = torch.eye(N, device="cuda")
        print(f"\n=== b{B} n{N} ===")

        torch.backends.cuda.matmul.allow_tf32 = False
        t_gram_fp32 = _t(lambda: torch.matmul(A.transpose(1, 2), A))
        torch.backends.cuda.matmul.allow_tf32 = True
        t_gram_tf32 = _t(lambda: torch.matmul(A.transpose(1, 2), A))
        torch.backends.cuda.matmul.allow_tf32 = False
        t_gram_bf16 = _t(lambda: torch.matmul(Abf.transpose(1, 2), Abf))

        M = (torch.matmul(A.transpose(1, 2), A) + 1e-3 * I).contiguous()
        t_chol = _t(lambda: torch.linalg.cholesky_ex(M))
        # serialization check: cost of ONE matrix's chol, x B (looped) -- if t_chol ~ this, serialized
        M1 = M[:1].contiguous()
        t_chol1 = _t(lambda: torch.linalg.cholesky_ex(M1))
        R = torch.linalg.cholesky_ex(M)[0].transpose(1, 2).contiguous()  # upper R
        t_solve = _t(lambda: torch.linalg.solve_triangular(R, A, upper=True, left=False))

        # bf16 A^T A accuracy (fp32 accumulate is default on tensor cores)
        M_ref = torch.matmul(A.transpose(1, 2), A)
        M_bf = torch.matmul(Abf.transpose(1, 2), Abf).float()
        rel = (M_bf - M_ref).norm() / M_ref.norm()

        one_iter = t_gram_bf16 + t_chol + t_solve
        print(f"  A^T A    fp32 {t_gram_fp32*1e3:7.1f}us   tf32 {t_gram_tf32*1e3:7.1f}us   bf16 {t_gram_bf16*1e3:7.1f}us")
        print(f"  chol(batched) {t_chol*1e3:7.1f}us   1-matrix x{B} = {t_chol1*B*1e3:7.1f}us"
              f"   {'<-- SERIALIZES' if t_chol > 0.5*t_chol1*B else '(batched OK)'}")
        print(f"  solve_tri {t_solve*1e3:7.1f}us")
        print(f"  ~1 CQR iter = gram+chol+solve = {one_iter*1e3:7.1f}us   -> CQR2 ~{2*one_iter*1e3:7.1f}us (+recon)")
        print(f"  bf16 A^T A rel-err = {rel:.2e}  (cond^2; gate factor-rtol ~{20*N*1.19e-7:.1e})")
