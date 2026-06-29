"""Custom modified-LU reconstruction kernel (single-block, SMEM-resident; n <= ~232).

Pipeline:  M=A^T A -> R_chol=chol(M)^T -> Q=A R_chol^-1   (torch / cuBLAS+cuSOLVER, fast)
           modified-LU on Q  (CUSTOM KERNEL) -> V (Householder vecs), tau, S (signs)
           R = S . R_chol  (= triu(Q~^T A));  H = V_below + triu(R)

Run on pod:  python dev/kqr.py
"""
import torch
from torch.utils.cpp_extension import load_inline
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check, cholesky_qr, SHAPES

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// One CUDA block per matrix. Working n x n matrix resident in dynamic shared memory.
// Performs the no-pivot "modified LU" of (Q - S): S diagonal sign matrix chosen per step,
// L (unit lower) = Householder vectors V, tau = 2/||v||^2 (=> householder_product is exactly
// orthogonal). Signs S returned so the host can form R = S . R_chol.
__global__ void modlu_kernel(const float* __restrict__ Q,
                             float* __restrict__ Hvecs,
                             float* __restrict__ tau,
                             float* __restrict__ sgn,
                             int n) {
    extern __shared__ float smem[];
    float* sQ   = smem;          // n*n
    float* ssig = smem + n * n;  // n

    int b = blockIdx.x;
    const float* Qb = Q + (size_t)b * n * n;
    int tid = threadIdx.x, nt = blockDim.x;

    for (int idx = tid; idx < n * n; idx += nt) sQ[idx] = Qb[idx];
    __syncthreads();

    for (int i = 0; i < n; ++i) {
        if (tid == 0) {
            float d  = sQ[i * n + i];
            float si = (d > 0.f) ? -1.f : 1.f;   // -sign(d); d==0 -> +1
            sQ[i * n + i] = d - si;              // pivot, |piv| in [1,2]
            ssig[i] = si;
        }
        __syncthreads();
        float piv = sQ[i * n + i];
        for (int j = i + 1 + tid; j < n; j += nt) sQ[j * n + i] /= piv;  // multipliers
        __syncthreads();
        int m = n - i - 1;
        for (int idx = tid; idx < m * m; idx += nt) {                    // trailing rank-1
            int j = i + 1 + idx / m, k = i + 1 + idx % m;
            sQ[j * n + k] -= sQ[j * n + i] * sQ[i * n + k];
        }
        __syncthreads();
    }

    for (int i = tid; i < n; i += nt) {
        float ss = 1.f;                          // ||v_i||^2 incl unit diagonal
        for (int j = i + 1; j < n; ++j) { float v = sQ[j * n + i]; ss += v * v; }
        tau[(size_t)b * n + i] = 2.f / ss;
        sgn[(size_t)b * n + i] = ssig[i];
    }
    for (int idx = tid; idx < n * n; idx += nt) {
        int r = idx / n, c = idx % n;
        Hvecs[(size_t)b * n * n + idx] = (r > c) ? sQ[idx] : 0.f;
    }
}

std::vector<torch::Tensor> modlu(torch::Tensor Q) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32 && Q.dim() == 3);
    Q = Q.contiguous();
    int batch = Q.size(0), n = Q.size(1);
    auto Hvecs = torch::zeros_like(Q);
    auto tau = torch::empty({batch, n}, Q.options());
    auto sgn = torch::empty({batch, n}, Q.options());
    size_t shmem = ((size_t)n * n + n) * sizeof(float);
    cudaFuncSetAttribute(modlu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    int threads = 256;
    modlu_kernel<<<batch, threads, shmem>>>(
        Q.data_ptr<float>(), Hvecs.data_ptr<float>(),
        tau.data_ptr<float>(), sgn.data_ptr<float>(), n);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    return {Hvecs, tau, sgn};
}
"""

CPP_SRC = "std::vector<torch::Tensor> modlu(torch::Tensor Q);"

_mod = load_inline(name="kqr_modlu", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                   functions=["modlu"], extra_cuda_cflags=["-O3"], verbose=False)


def reconstruct_lu_faithful(Q, R_chol):
    """Pure-torch mirror of the kernel (validates the algorithm incl. the S.R_chol shortcut)."""
    b, n, _ = Q.shape
    B = Q.clone()
    V = torch.zeros_like(Q)
    S = torch.empty(b, n, device=Q.device)
    for i in range(n):
        d = B[:, i, i]
        si = torch.where(d > 0, -torch.ones_like(d), torch.ones_like(d))
        piv = d - si
        B[:, i, i] = piv
        S[:, i] = si
        V[:, i, i] = 1.0
        if i + 1 < n:
            col = B[:, i + 1:, i] / piv.unsqueeze(-1)
            V[:, i + 1:, i] = col
            B[:, i + 1:, i + 1:] -= col.unsqueeze(-1) * B[:, i, i + 1:].unsqueeze(1)
    tau = 2.0 / (torch.tril(V, -1).pow(2).sum(1) + 1.0)
    R = S.unsqueeze(-1) * R_chol
    H = torch.tril(V, -1) + torch.triu(R)
    return H, tau


EPS32 = torch.finfo(torch.float32).eps


def _chol_with_shift(M):
    """Cholesky upper R (M=R^T R); shift the matrices that fail to keep them in the fast path."""
    n = M.shape[-1]
    L, info = torch.linalg.cholesky_ex(M)
    bad = info != 0
    if bad.any():
        dmax = M.diagonal(dim1=-2, dim2=-1).amax(-1)            # ~||M||
        shift = (11.0 * n * EPS32) * dmax.clamp_min(1e-30)
        I = torch.eye(n, device=M.device)
        Mb = M[bad] + shift[bad].view(-1, 1, 1) * I
        Lb, infob = torch.linalg.cholesky_ex(Mb)
        L = L.clone(); L[bad] = Lb
        info = info.clone(); info[bad] = infob
    return L.mT, info == 0


def robust_cqr(A, passes=2):
    """Shifted CholeskyQR{passes}: returns Q, accumulated R (A=QR), per-matrix ok mask."""
    n = A.shape[-1]
    M = A.mT @ A
    R, ok = _chol_with_shift(M)
    Q = torch.linalg.solve_triangular(R, A, upper=True, left=False)
    for _ in range(passes - 1):
        M2 = Q.mT @ Q
        R2, ok2 = _chol_with_shift(M2)
        Q = torch.linalg.solve_triangular(R2, Q, upper=True, left=False)
        R = R2 @ R
        ok = ok & ok2
    return Q, R, ok


def pipeline(A, passes=2, use_kernel=True, gate_mult=4.0):
    n = A.shape[-1]
    Q, R_chol, ok = robust_cqr(A, passes=passes)
    if use_kernel:
        Hvecs, tau, S = _mod.modlu(Q)
        R = S.unsqueeze(-1) * R_chol
        H = Hvecs + torch.triu(R)
    else:
        H, tau = reconstruct_lu_faithful(Q, R_chol)
    # theta-gate: route any matrix whose cheap factorization is off-tolerance to geqrf.
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
    from oracle import make_batch as mb
    print("=== dense shapes that fit the single-block kernel (n<=224) ===")
    for (b, n, cond, case) in SHAPES:
        if case != "dense" or n > 224:
            continue
        A = make_batch(b, n, cond, case, seed=0)
        H, tau = pipeline(A, use_kernel=True)
        fr, og, ft, ot, ps = check(A, H, tau)
        print(f"  b{b}_n{n}_{case}: factor={fr:.2e}/{ft:.2e} ortho={og:.2e}/{ot:.2e} "
              f"pass={int(ps.sum())}/{len(ps)} {'OK' if ps.all() else 'FAIL'}")

    print("=== stress structures at kernel-sized n (robust path) ===")
    for case in ["rankdef", "clustered", "nearrank", "mixed"]:
        b, n = 40, 176
        A = mb(b, n, 2, case, seed=3)
        H, tau = pipeline(A, use_kernel=True)
        fr, og, ft, ot, ps = check(A, H, tau)
        print(f"  b{b}_n{n}_{case}: factor={fr:.2e}/{ft:.2e} ortho={og:.2e}/{ot:.2e} "
              f"pass={int(ps.sum())}/{len(ps)} {'OK' if ps.all() else 'FAIL'}")


if __name__ == "__main__":
    main()
