"""Phase-1 FULLY-FUSED single-block QR kernel (n that fits one block's SMEM).

Everything in ONE custom CUDA kernel, one block per matrix, no cuBLAS/cuSOLVER in the
compute path:

    A  --Gram(WMMA bf16xORDER, sm_100)-->  M1 = A^T A
       --in-SMEM Cholesky-------------->    R1 = chol(M1)^T (upper)
       --in-SMEM TRSM------------------>    Q1 = A R1^{-1}
       (CQR2: repeat on Q1 -> R2, Q2)
       --in-SMEM modified-LU----------->    L (Householder vecs), tau, S (signs)
       --FACTOR CLOSURE---------------->    R_out = S . (Q2^T A)   (recomputed from the
                                            clean CQR2 Q, NOT from R_chol)
       write H = tril(L,-1) + triu(R_out),  tau

TWO-BUFFER FACTOR-CLOSURE layout (lifted from the parallel claude.ai thread + our review):
keep Q2 live in `bufA` through the modified-LU (run modlu on a COPY in `bufM`), then rebuild
R from the clean Q via R_out = S.(Q2^T A). Benefits over the old `R = S.R_chol`:
  * only 2 n*n FP32 buffers (raises the single-block n ceiling), and
  * the factor residual stays clean even with a bf16 Gram (R is rebuilt from the
    CQR2-corrected Q, so it is NOT tied to the bf16-polluted R_chol). With an emulated Gram,
    `R = S.R_chol` would inherit the Gram's bf16 error; closure does not.
See memory/qr-fused-precision-and-cluster-budget.md.

PRECISION (the kappa^2 finding, numerically validated in the parallel thread): an emulated
Gram needs CQR2 (single pass fails orthogonality -- kappa^2 amplifies the bf16 error past the
gate). bf16x3 (order=1) is good to ~n<=512; n>=1024 needs bf16x6 (order=2). The `order` knob
below selects 3 vs 6 limb cross-terms at launch so we can sweep it on the pod. Our Phase-1
n<=176 target is safe on order=1+CQR2.

SCOPE (the SMEM wall): one block has ~227 KB dynamic SMEM on sm_100. With 2 n*n FP32 buffers
(+3 n*n bf16 limb planes for the WMMA Gram) the single-block path tops out around:
    n <= ~168 with the FP32 reference Gram, n <= ~127 with the WMMA limbs.
n=176 needs per-tile limb staging (build limbs inside the WMMA k-loop instead of 3 full
planes) -- a follow-up. n >= 256 (incl. 352, 512) does NOT fit one block -> cluster/DSMEM.

GRAM_MODE (compile flag): 0 = plain FP32 FMA Gram (reference, for clean bring-up),
                          3 = bf16x{3,6} emulated-FP32 WMMA on sm_100 (target; order chooses).
We build BOTH and validate mode 3 against mode 0 + the oracle checker.

Run on the Runpod B200:  python dev/kqr_fused.py
"""
import torch
from torch.utils.cpp_extension import load_inline
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check
from kqr import robust_cqr  # for the pure-torch cross-check mirror

torch.backends.cuda.matmul.allow_tf32 = False
EPS32 = torch.finfo(torch.float32).eps

# --------------------------------------------------------------------------- #
# CUDA source. GRAM_MODE is injected via -DGRAM_MODE=0/3 at compile time;      #
# `order` (limb cross-terms, WMMA only) is a runtime kernel argument.          #
# --------------------------------------------------------------------------- #
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#ifndef GRAM_MODE
#define GRAM_MODE 3            /* 0 = FP32 reference, 3 = bf16 WMMA (sm_100) */
#endif
#ifndef CQR_PASSES
#define CQR_PASSES 2           /* CQR2 is mandatory with an emulated Gram (kappa^2) */
#endif

#if GRAM_MODE==3
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
#endif

/* ============================ stage device fns ============================ */

/* Gram (FP32 reference): M = X^T X, X and M are n x n row-major in SMEM. */
__device__ void gram_fp32(const float* __restrict__ X, float* __restrict__ M, int n) {
    int tid = threadIdx.x, nt = blockDim.x;
    for (int idx = tid; idx < n * n; idx += nt) {
        int i = idx / n, j = idx % n;
        if (i <= j) {
            float s = 0.f;
            for (int k = 0; k < n; ++k) s += X[k * n + i] * X[k * n + j];
            M[i * n + j] = s;
            M[j * n + i] = s;
        }
    }
    __syncthreads();
}

#if GRAM_MODE==3
/* Split each FP32 element of X into 3 bf16 limbs (hi, mid, lo) ~= 24-bit mantissa. */
__device__ void split_limbs(const float* __restrict__ X, __nv_bfloat16* Lh,
                            __nv_bfloat16* Lm, __nv_bfloat16* Ll, int n) {
    int tid = threadIdx.x, nt = blockDim.x;
    for (int idx = tid; idx < n * n; idx += nt) {
        float x  = X[idx];
        __nv_bfloat16 h = __float2bfloat16_rn(x);
        float r1 = x  - __bfloat162float(h);
        __nv_bfloat16 m = __float2bfloat16_rn(r1);
        float r2 = r1 - __bfloat162float(m);
        __nv_bfloat16 l = __float2bfloat16_rn(r2);
        Lh[idx] = h; Lm[idx] = m; Ll[idx] = l;
    }
    __syncthreads();
}

/* M (+)= A^T B for bf16 limb buffers A,B (n x n row-major, n % 16 == 0). One warp per
   16x16 output tile, grid-strided over warps. init=true overwrites M, else accumulates. */
__device__ void gemm_AtB_bf16_acc(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B,
                                  float* __restrict__ M, int n, bool init) {
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    int T = n / 16;
    for (int t = warp; t < T * T; t += nwarps) {
        int ti = t / T, tj = t % T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.0f);
        for (int kt = 0; kt < T; ++kt) {
            /* a = A^T tile (col_major load of A gives the transpose); b = B tile. */
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
            wmma::load_matrix_sync(af, A + (kt * 16) * n + ti * 16, n);
            wmma::load_matrix_sync(bf, B + (kt * 16) * n + tj * 16, n);
            wmma::mma_sync(cf, af, bf, cf);
        }
        float* Mt = M + (ti * 16) * n + tj * 16;
        if (init) {
            wmma::store_matrix_sync(Mt, cf, n, wmma::mem_row_major);
        } else {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> ex;
            wmma::load_matrix_sync(ex, Mt, n, wmma::mem_row_major);
            for (int e = 0; e < cf.num_elements; ++e) cf.x[e] += ex.x[e];
            wmma::store_matrix_sync(Mt, cf, n, wmma::mem_row_major);
        }
    }
    __syncthreads();
}

/* bf16-emulated FP32 Gram. order=1 -> bf16x3 (~3.6e-6, good to ~n<=512);
   order=2 -> bf16x6 (~5e-7, needed for n>=1024).
     order>=1 (x3):  h^Th + (h^Tm + m^Th)
     order>=2 (x6): +(h^Tl + l^Th + m^Tm) */
__device__ void gram_bf16(const float* __restrict__ X, float* __restrict__ M, int n,
                          int order, __nv_bfloat16* Lh, __nv_bfloat16* Lm, __nv_bfloat16* Ll) {
    split_limbs(X, Lh, Lm, Ll, n);
    gemm_AtB_bf16_acc(Lh, Lh, M, n, true);     /* h0^T h0 */
    gemm_AtB_bf16_acc(Lh, Lm, M, n, false);    /* h0^T h1 */
    gemm_AtB_bf16_acc(Lm, Lh, M, n, false);    /* h1^T h0  -> bf16x3 */
    if (order >= 2) {
        gemm_AtB_bf16_acc(Lh, Ll, M, n, false);  /* h0^T h2 */
        gemm_AtB_bf16_acc(Ll, Lh, M, n, false);  /* h2^T h0 */
        gemm_AtB_bf16_acc(Lm, Lm, M, n, false);  /* h1^T h1 -> bf16x6 */
    }
}
#endif  /* GRAM_MODE==3 */

/* In-place row-Cholesky: M(upper) -> R(upper) with M = R^T R. Guards non-positive
   pivots to a tiny value (no NaN/Inf); the host theta-gate routes those to geqrf. */
__device__ void chol_upper(float* __restrict__ M, int n) {
    int tid = threadIdx.x, nt = blockDim.x;
    for (int j = 0; j < n; ++j) {
        if (tid == 0) {
            float s = M[j * n + j];
            for (int k = 0; k < j; ++k) { float r = M[k * n + j]; s -= r * r; }
            M[j * n + j] = (s > 1e-30f) ? sqrtf(s) : 1e-15f;
        }
        __syncthreads();
        float rjj = M[j * n + j];
        for (int i = j + 1 + tid; i < n; i += nt) {
            float s = M[j * n + i];
            for (int k = 0; k < j; ++k) s -= M[k * n + j] * M[k * n + i];
            M[j * n + i] = s / rjj;
        }
        __syncthreads();
    }
}

/* In-place right triangular solve X <- X R^{-1}, R upper. One thread per row of X. */
__device__ void trsm_right_upper(float* __restrict__ X, const float* __restrict__ R, int n) {
    int tid = threadIdx.x, nt = blockDim.x;
    for (int r = tid; r < n; r += nt) {
        float* xr = X + r * n;
        for (int c = 0; c < n; ++c) {
            float s = xr[c];
            for (int k = 0; k < c; ++k) s -= xr[k] * R[k * n + c];
            xr[c] = s / R[c * n + c];
        }
    }
    __syncthreads();
}

/* Interleaved modified-LU of (B - S) = L U, in place on B (= orthonormal Q). Emits
   L = Householder vectors (below diag of B), signs S, tau = 2/||v||^2 (=> the rebuilt
   reflectors are EXACTLY orthogonal). Identical to the shipped `modlu` kernel core. */
__device__ void modlu_inplace(float* __restrict__ B, float* __restrict__ sgn,
                              float* __restrict__ tau_b, int n) {
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = 0; i < n; ++i) {
        if (tid == 0) {
            float d = B[i * n + i];
            float s = (d > 0.f) ? -1.f : 1.f;       /* -sign(d); d==0 -> +1 */
            B[i * n + i] = d - s;                    /* pivot, |piv| in [1,2] */
            sgn[i] = s;
        }
        __syncthreads();
        float piv = B[i * n + i];
        for (int j = i + 1 + tid; j < n; j += nt) B[j * n + i] /= piv;   /* multipliers */
        __syncthreads();
        int m = n - i - 1;
        for (int idx = tid; idx < m * m; idx += nt) {                    /* trailing rank-1 */
            int j = i + 1 + idx / m, k = i + 1 + idx % m;
            B[j * n + k] -= B[j * n + i] * B[i * n + k];
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += nt) {              /* tau (reads V below diag) */
        float ss = 1.f;
        for (int j = i + 1; j < n; ++j) { float v = B[j * n + i]; ss += v * v; }
        tau_b[i] = 2.f / ss;
    }
    __syncthreads();
}

/* ============================ fused kernel ============================ */
__global__ void fused_qr_kernel(const float* __restrict__ Ain, float* __restrict__ Hout,
                                float* __restrict__ tauout, int n, int order) {
    extern __shared__ float smem[];
    float* bufA = smem;              /* A -> Q1 -> Q2 (kept live)   (n*n) */
    float* bufM = bufA + n * n;      /* M -> R, then Q2-copy -> L   (n*n) */
    float* sgn  = bufM + n * n;      /* signs                       (n)   */
    float* pad  = sgn + n;           /* (n) reserved / alignment           */
#if GRAM_MODE==3
    __nv_bfloat16* Lh = (__nv_bfloat16*)(pad + n);
    __nv_bfloat16* Lm = Lh + n * n;
    __nv_bfloat16* Ll = Lm + n * n;
#endif
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float* Ab = Ain + (size_t)b * n * n;       /* global A (row-major), kept for closure */
    for (int idx = tid; idx < n * n; idx += nt) bufA[idx] = Ab[idx];
    __syncthreads();

#if GRAM_MODE==3
    #define GRAM(X, M) gram_bf16((X), (M), n, order, Lh, Lm, Ll)
#else
    #define GRAM(X, M) gram_fp32((X), (M), n)
#endif

    /* CQR pass 1: M1=A^T A -> R1=chol -> Q1 = A R1^{-1} (overwrites bufA) */
    GRAM(bufA, bufM);
    chol_upper(bufM, n);
    trsm_right_upper(bufA, bufM, n);

#if CQR_PASSES>=2
    /* CQR pass 2: M2=Q1^T Q1 -> R2=chol (overwrites R1, not needed) -> Q2 = Q1 R2^{-1} */
    GRAM(bufA, bufM);
    chol_upper(bufM, n);
    trsm_right_upper(bufA, bufM, n);
#endif
    /* bufA = Q2 (orthonormal, kept live); bufM = R2 (discardable). */

    /* modified-LU on a COPY of Q2 so bufA stays intact for the factor closure. */
    for (int idx = tid; idx < n * n; idx += nt) bufM[idx] = bufA[idx];
    __syncthreads();
    modlu_inplace(bufM, sgn, tauout + (size_t)b * n, n);   /* bufM below-diag = L */

    /* assemble H: below diag = L; on/above diag = R_out = S .* (Q2^T A). */
    float* Hb = Hout + (size_t)b * n * n;
    for (int idx = tid; idx < n * n; idx += nt) {
        int i = idx / n, j = idx % n;
        if (i > j) {
            Hb[idx] = bufM[i * n + j];                       /* Householder vec component */
        } else {
            float acc = 0.f;                                 /* (Q2^T A)[i,j] */
            for (int k = 0; k < n; ++k) acc += bufA[k * n + i] * Ab[k * n + j];
            Hb[idx] = sgn[i] * acc;                          /* sign-corrected R */
        }
    }
    #undef GRAM
}

std::vector<torch::Tensor> fused_qr(torch::Tensor A, int order) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32 && A.dim() == 3);
    A = A.contiguous();
    int batch = A.size(0), n = A.size(1);
#if GRAM_MODE==3
    TORCH_CHECK(n % 16 == 0, "bf16 WMMA Gram requires n divisible by 16");
#endif
    auto H = torch::empty_like(A);
    auto tau = torch::empty({batch, n}, A.options());
    size_t smem = (size_t)(2 * n * n + 2 * n) * sizeof(float);
#if GRAM_MODE==3
    smem += (size_t)(3 * n * n) * sizeof(__nv_bfloat16);
#endif
    cudaFuncSetAttribute(fused_qr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    fused_qr_kernel<<<batch, 256, smem>>>(A.data_ptr<float>(), H.data_ptr<float>(),
                                          tau.data_ptr<float>(), n, order);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
    return {H, tau};
}
"""
CPP_SRC = "std::vector<torch::Tensor> fused_qr(torch::Tensor A, int order);"


def _build(gram_mode):
    flags = ["-O3", f"-DGRAM_MODE={gram_mode}"]
    if gram_mode == 3:
        flags += ["-gencode=arch=compute_100,code=sm_100"]
    return load_inline(name=f"kqr_fused_g{gram_mode}", cpp_sources=[CPP_SRC],
                       cuda_sources=[CUDA_SRC], functions=["fused_qr"],
                       extra_cuda_cflags=flags, verbose=True)


# --------------------------------------------------------------------------- #
# Pure-torch mirror of the kernel's exact arithmetic (factor closure variant). #
# --------------------------------------------------------------------------- #
def fused_qr_mirror(A, passes=2):
    n = A.shape[-1]
    Q, _, _ = robust_cqr(A, passes=passes)              # Q2 (orthonormal)
    b = A.shape[0]
    B = Q.contiguous().clone()
    V = torch.zeros_like(Q)
    S = torch.empty(b, n, device=A.device)
    for i in range(n):
        d = B[:, i, i]
        si = torch.where(d > 0, -torch.ones_like(d), torch.ones_like(d))
        piv = d - si
        B[:, i, i] = piv; S[:, i] = si; V[:, i, i] = 1.0
        if i + 1 < n:
            col = B[:, i + 1:, i] / piv.unsqueeze(-1)
            V[:, i + 1:, i] = col
            B[:, i + 1:, i + 1:] -= col.unsqueeze(-1) * B[:, i, i + 1:].unsqueeze(1)
    tau = 2.0 / (torch.tril(V, -1).pow(2).sum(1) + 1.0)
    Rout = S.unsqueeze(-1) * (Q.mT @ A)                 # factor closure: S .* (Q^T A)
    H = torch.tril(V, -1) + torch.triu(Rout)
    return H, tau


# --------------------------------------------------------------------------- #
# Validation driver (run on the B200).                                         #
# --------------------------------------------------------------------------- #
def _report(tag, A, H, tau):
    fr, og, ft, ot, ps = check(A, H, tau)
    flag = "OK " if ps.all() else "FAIL"
    print(f"  {tag:24s} factor={fr:.2e}/{ft:.2e}  ortho={og:.2e}/{ot:.2e}  "
          f"pass={int(ps.sum())}/{len(ps)} {flag}")
    return bool(ps.all())


def main():
    print("Phase-1 fused single-block kernel (2-buffer factor-closure) -- validation\n")

    # mode 0 (FP32 Gram) proves the fused pipeline; 2 FP32 buffers fit n up to ~168.
    print("--- GRAM_MODE=0 (FP32 reference Gram) ---")
    k0 = _build(0)
    for (b, n, cond) in [(64, 32, 1), (64, 32, 2), (32, 64, 2), (16, 128, 2)]:
        A = make_batch(b, n, cond, "dense", seed=0)
        H, tau = k0.fused_qr(A, 1)            # order ignored in FP32 mode
        _report(f"n{n}_c{cond}", A, H, tau)

    # mode 3 (bf16 WMMA, sm_100). 3 limb planes -> fits n up to ~127. Sweep order 1 (x3) / 2 (x6).
    print("\n--- GRAM_MODE=3 (bf16 WMMA, sm_100) ---")
    k3 = _build(3)
    for (b, n, cond) in [(64, 32, 1), (64, 32, 2), (32, 64, 2)]:
        A = make_batch(b, n, cond, "dense", seed=0)
        for order in (1, 2):                  # 1 = bf16x3, 2 = bf16x6
            H, tau = k3.fused_qr(A, order)
            _report(f"n{n}_c{cond}_x{3 if order==1 else 6}", A, H, tau)
        Hm, tm = fused_qr_mirror(A)           # cross-check vs torch twin
        _report(f"n{n}_c{cond}.mirror", A, Hm, tm)

    print("\nNext: per-tile limb staging to reach n=176, then the cluster/DSMEM path for "
          "n>=256 (8-block cluster holds n=512's 1 MB working set).")


if __name__ == "__main__":
    main()
