"""Probe whether the cuBLAS BF16x9 FP32-emulation back-door actually engages on this B200.

Background: torch.backends.cuda.matmul.fp32_precision='bf16x9' is a documented NO-OP (torch's
enum only knows ieee/tf32). The real switch is the cuBLAS env back-door, read at cuBLAS init:
    CUBLAS_EMULATE_SINGLE_PRECISION=1   CUBLAS_EMULATION_STRATEGY=performant|eager
gated to sm_100/103 + CUDA >= 12.9. This times representative FP32 GEMMs and prints an FP64
checksum of the result. Run the SAME script with and without the env vars and compare:
  * ENGAGED  -> time changes AND checksum shifts in low digits (bf16x9 ~ fp32 to ~1e-6)
  * NO-OP    -> identical time and identical checksum (strategy declined or unsupported)

Run on pod:
    source /workspace/qr/env.sh
    python dev/probe_bf16x9.py
    CUBLAS_EMULATE_SINGLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=performant python dev/probe_bf16x9.py
    CUBLAS_EMULATE_SINGLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager      python dev/probe_bf16x9.py
"""
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = False   # keep TF32 out of the comparison
DEV = "cuda"


def t_ms(fn, it=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def main():
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  dev {torch.cuda.get_device_name(0)}")
    print(f"CUBLAS_EMULATE_SINGLE_PRECISION={os.environ.get('CUBLAS_EMULATE_SINGLE_PRECISION')}  "
          f"CUBLAS_EMULATION_STRATEGY={os.environ.get('CUBLAS_EMULATION_STRATEGY')}")
    torch.manual_seed(0)
    print(f"  {'shape':18s} {'ms':>9s}   {'fp64 checksum(|C|)':>22s}")

    for tag, m, k, n in [("sq2048", 2048, 2048, 2048), ("sq4096", 4096, 4096, 4096)]:
        A = torch.randn(m, k, device=DEV)
        B = torch.randn(k, n, device=DEV)
        ms = t_ms(lambda: A @ B)
        cs = (A @ B).double().abs().sum().item()
        print(f"  {tag:18s} {ms:9.3f}   {cs:22.10e}")

    # the contest's dominant Gram: batched A^T A, b=640 n=512
    b, n = 640, 512
    A = torch.randn(b, n, n, device=DEV)
    ms = t_ms(lambda: A.mT @ A)
    cs = (A.mT @ A).double().abs().sum().item()
    print(f"  {'gram_b640_n512':18s} {ms:9.3f}   {cs:22.10e}")


if __name__ == "__main__":
    main()
