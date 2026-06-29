"""Blocked interleaved modified-LU for large n (n > ~232 doesn't fit a single block).

Right-looking blocked LU:
  per panel [p0:pe]:
    (1) panel-factor kernel  -> interleaved modified-LU on tall panel B[p0:n, p0:pe] in SMEM
                                (custom CUDA; produces L multipliers + signs S)
    (2) TRSM (torch/cuBLAS)  -> U rows  B[p0:pe, pe:] = L_diag^-1 B[p0:pe, pe:]
    (3) GEMM (torch/cuBLAS)  -> trailing B[pe:, pe:] -= L_below @ U_rows   (<- emulated-FP32 later)
  end:  V = tril(B,-1) = Householder vectors;  tau = 2/||v||^2;  R = S . R_chol
Run on pod:  python dev/kqr_blocked.py
"""
import torch
from torch.utils.cpp_extension import load_inline
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check
from kqr import robust_cqr

torch.backends.cuda.matmul.allow_tf32 = False
EPS32 = torch.finfo(torch.float32).eps

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// One block per matrix. Unblocked interleaved modified-LU of the tall panel
// B[p0:n, p0:pe] (rows = n-p0, cols = nb = pe-p0), resident in dynamic shared memory.
__global__ void panel_factor_kernel(float* __restrict__ B, float* __restrict__ S,
                                    int n, int p0, int nb) {
    extern __shared__ float sP[];           // rows * nb
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int rows = n - p0;
    float* Bb = B + (size_t)b * n * n;
    // load tall panel: sP[r*nb + c] = B[p0+r, p0+c]
    for (int idx = tid; idx < rows * nb; idx += nt) {
        int r = idx / nb, c = idx % nb;
        sP[idx] = Bb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    __syncthreads();

    for (int i = 0; i < nb; ++i) {
        if (tid == 0) {
            float d  = sP[i * nb + i];
            float si = (d > 0.f) ? -1.f : 1.f;
            sP[i * nb + i] = d - si;
            S[(size_t)b * n + (p0 + i)] = si;
        }
        __syncthreads();
        float piv = sP[i * nb + i];
        for (int j = i + 1 + tid; j < rows; j += nt) sP[j * nb + i] /= piv;
        __syncthreads();
        int mr = rows - i - 1, mc = nb - i - 1;
        for (int idx = tid; idx < mr * mc; idx += nt) {       // within-panel trailing
            int j = i + 1 + idx / mc, k = i + 1 + idx % mc;
            sP[j * nb + k] -= sP[j * nb + i] * sP[i * nb + k];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < rows * nb; idx += nt) {          // write back
        int r = idx / nb, c = idx % nb;
        Bb[(size_t)(p0 + r) * n + (p0 + c)] = sP[idx];
    }
}

void panel_factor(torch::Tensor B, torch::Tensor S, int p0, int nb) {
    int batch = B.size(0), n = B.size(1);
    int rows = n - p0;
    size_t shmem = (size_t)rows * nb * sizeof(float);
    cudaFuncSetAttribute(panel_factor_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_factor_kernel<<<batch, 256, shmem>>>(
        B.data_ptr<float>(), S.data_ptr<float>(), n, p0, nb);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}
"""

CPP_SRC = "void panel_factor(torch::Tensor B, torch::Tensor S, int p0, int nb);"

_blk = load_inline(name="kqr_panel", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                   functions=["panel_factor"], extra_cuda_cflags=["-O3"], verbose=False)


def blocked_modlu(Q):
    """Returns Householder vectors V (below diag), tau, signs S — all batched."""
    b, n, _ = Q.shape
    B = Q.contiguous().clone()
    S = torch.empty(b, n, device=Q.device)
    nb0 = max(8, min(64, 32768 // n))            # keep tall panel <= ~128KB SMEM
    p0 = 0
    while p0 < n:
        nb = min(nb0, n - p0)
        pe = p0 + nb
        _blk.panel_factor(B, S, p0, nb)          # (1) panel factor -> L, signs
        if pe < n:
            Ldiag = B[:, p0:pe, p0:pe]           # unit-lower (implicit unit diag)
            rhs = B[:, p0:pe, pe:]
            U = torch.linalg.solve_triangular(Ldiag, rhs, upper=False,
                                              unitriangular=True, left=True)   # (2) TRSM
            B[:, p0:pe, pe:] = U
            B[:, pe:, pe:] -= B[:, pe:, p0:pe] @ U                            # (3) GEMM
        p0 = pe
    V = torch.tril(B, -1)
    tau = 2.0 / (V.pow(2).sum(1) + 1.0)
    return V, tau, S


def pipeline(A, passes=2, gate_mult=4.0):
    n = A.shape[-1]
    Q, R_chol, ok = robust_cqr(A, passes=passes)
    V, tau, S = blocked_modlu(Q)
    R = S.unsqueeze(-1) * R_chol
    H = V + torch.triu(R)
    I = torch.eye(n, device=A.device).expand_as(Q)
    defect = torch.linalg.matrix_norm((Q.mT @ Q - I).float(), ord=1, dim=(-2, -1))
    good = ok & torch.isfinite(defect) & (defect < gate_mult * 100 * n * EPS32)
    bad = ~good
    if bad.any():
        Hb, taub = torch.geqrf(A[bad])
        H = H.clone(); tau = tau.clone()
        H[bad], tau[bad] = Hb, taub
    return H, tau


def main():
    print("=== blocked modified-LU over large-n shapes ===")
    tests = [(40, 352, 1, "dense"), (64, 512, 2, "dense"), (20, 1024, 2, "dense"),
             (40, 512, 0, "clustered"), (40, 512, 0, "rankdef"), (20, 1024, 0, "nearrank"),
             (40, 512, 2, "mixed")]
    for (b, n, cond, case) in tests:
        A = make_batch(b, n, cond, case, seed=0)
        H, tau = pipeline(A)
        fr, og, ft, ot, ps = check(A, H, tau)
        flag = "OK" if ps.all() else "FAIL"
        print(f"  b{b}_n{n}_{case:9s}: factor={fr:.2e}/{ft:.2e} ortho={og:.2e}/{ot:.2e} "
              f"pass={int(ps.sum())}/{len(ps)} {flag}")


if __name__ == "__main__":
    main()
