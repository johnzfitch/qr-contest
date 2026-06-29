"""Runpod B200 environment probe — run FIRST on the pod.

Confirms sm_100 + nvcc + the torch APIs the pipeline depends on, before we build
any kernels. Usage:  python dev/probe.py
"""
import subprocess, sys

print("=== torch / CUDA ===")
import torch
print("torch        :", torch.__version__)
print("cuda (torch) :", torch.version.cuda)
print("cuda avail   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       :", torch.cuda.get_device_name(0))
    print("capability   :", torch.cuda.get_device_capability(0), "(want (10, 0) for B200/sm_100)")
    print("bf16 support :", torch.cuda.is_bf16_supported())

print("\n=== nvcc ===")
try:
    print(subprocess.check_output(["nvcc", "--version"]).decode().strip())
except Exception as e:
    print("nvcc NOT found:", e)

print("\n=== API checks (the pipeline depends on these) ===")
dev = "cuda" if torch.cuda.is_available() else "cpu"
A = torch.randn(4, 16, 16, device=dev, dtype=torch.float32)
for name, fn in [
    ("geqrf (batched)",      lambda: torch.geqrf(A)),
    ("householder_product",  lambda: torch.linalg.householder_product(*torch.geqrf(A))),
    ("cholesky_ex (batched)",lambda: torch.linalg.cholesky_ex(A.mT @ A + torch.eye(16, device=dev))),
    ("solve_triangular",     lambda: torch.linalg.solve_triangular(
                                 torch.triu(A[0]), A[0], upper=True, left=False)),
    ("matrix_norm ord=1",    lambda: torch.linalg.matrix_norm(A, ord=1, dim=(-2, -1))),
]:
    try:
        out = fn()
        shp = tuple(out.shape) if torch.is_tensor(out) else type(out).__name__
        print(f"  OK   {name:24s} -> {shp}")
    except Exception as e:
        print(f"  FAIL {name:24s} -> {e}")

print("\nProbe done.")
