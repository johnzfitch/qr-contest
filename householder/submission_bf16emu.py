#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200
"""Blocked WY-Householder QR (qr_v2) — cuBLAS fp32-EMULATION trailing probe (#5).

IDENTICAL compute to submission_full.py. The ONLY change: enable the cuBLAS
single-precision-emulation back-door (set BEFORE torch/cuBLAS init) so the fp32
compact-WY trailing GEMMs run as bf16x9 limbs on tensor cores — numerically fp32
(clears the per-column gate that single-limb TF32 failed on band/rowscale/mixed),
but at tensor-core throughput instead of fp32 CUDA cores. Forkless: same path,
same precision, every matrix. Diagnostic: big geomean drop => trailing was the
wall; small drop => the scalar panel_geqrt kernel is now the wall (=> EG panel).
Back-door is sm_100 + CUDA>=12.9 gated (this B200 qualifies); a documented no-op
elsewhere, so it degrades to plain fp32 rather than breaking.
"""
import os
os.environ.setdefault("CUBLAS_EMULATE_SINGLE_PRECISION", "1")
os.environ.setdefault("CUBLAS_EMULATION_STRATEGY", "performant")

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False   # fp32 semantics; cuBLAS emulates on TC

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

/* block-wide sum reduction of `val`; result broadcast via sred[0]. blockDim.x a power of 2. */
__device__ float block_sum(float val, float* sred) {
    int tid = threadIdx.x, nt = blockDim.x;
    sred[tid] = val;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
    }
    float r = sred[0];
    __syncthreads();
    return r;
}

/* One block per matrix. Factor the tall panel B[p0:n, p0:pe] (m=n-p0 rows, nb cols) by
   unblocked Householder geqr2, then build the nb x nb compact-WY T factor (DLARFT). */
__global__ void panel_geqrt_kernel(float* __restrict__ B, float* __restrict__ Tg,
                                   float* __restrict__ taug, int n, int p0, int nb) {
    extern __shared__ float smem[];
    int m = n - p0;
    float* sP   = smem;                 /* m*nb  panel (row-major sP[r*nb+c]) */
    float* sT   = sP + (size_t)m * nb;  /* nb*nb T (row-major) */
    float* stau = sT + (size_t)nb * nb; /* nb    tau */
    float* sz   = stau + nb;            /* nb    DLARFT workspace */
    float* sred = sz + nb;              /* blockDim.x  reduction scratch */
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    float* Bb = B + (size_t)b * n * n;

    for (int idx = tid; idx < m * nb; idx += nt) {       /* load panel */
        int r = idx / nb, c = idx % nb;
        sP[idx] = Bb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < nb * nb; idx += nt) sT[idx] = 0.f;   /* T lower-tri stays 0 */
    __syncthreads();

    /* ---- unblocked geqr2 over the nb panel columns ---- */
    __shared__ float s_tau, s_denom;
    for (int j = 0; j < nb; ++j) {
        float part = 0.f;                                /* ||tail||^2 of column j */
        for (int r = j + 1 + tid; r < m; r += nt) { float v = sP[r * nb + j]; part += v * v; }
        float xn2 = block_sum(part, sred);
        if (tid == 0) {
            float alpha = sP[j * nb + j];
            float beta, tauj, denom;
            if (xn2 > 0.f) {                              /* DLARFG */
                float normf = sqrtf(alpha * alpha + xn2);
                beta  = (alpha >= 0.f) ? -normf : normf;
                tauj  = (beta - alpha) / beta;
                denom = alpha - beta;
            } else { beta = alpha; tauj = 0.f; denom = 1.f; }   /* already triangular */
            sP[j * nb + j] = beta;
            s_tau = tauj; s_denom = denom; stau[j] = tauj;
        }
        __syncthreads();
        float denom = s_denom, tauj = s_tau;
        for (int r = j + 1 + tid; r < m; r += nt) sP[r * nb + j] /= denom;   /* v = tail/denom */
        __syncthreads();
        /* apply H_j to trailing panel columns c in (j, nb): one thread per column */
        for (int c = j + 1 + tid; c < nb; c += nt) {
            float w = sP[j * nb + c];                    /* vfull[j] = 1 */
            for (int r = j + 1; r < m; ++r) w += sP[r * nb + j] * sP[r * nb + c];
            w *= tauj;
            sP[j * nb + c] -= w;
            for (int r = j + 1; r < m; ++r) sP[r * nb + c] -= sP[r * nb + j] * w;
        }
        __syncthreads();
    }

    /* ---- DLARFT: compact-WY T (nb x nb upper-tri), forward columnwise ---- */
    for (int j = 0; j < nb; ++j) {
        if (j > 0) {
            for (int i = tid; i < j; i += nt) {          /* z[i] = -tau_j * (V[:,i].V[:,j]) */
                float d = sP[j * nb + i];                 /* r=j term: V[j,i]*1 */
                for (int r = j + 1; r < m; ++r) d += sP[r * nb + i] * sP[r * nb + j];
                sz[i] = -stau[j] * d;
            }
            __syncthreads();
            for (int i = tid; i < j; i += nt) {          /* T[0:j,j] = T[0:j,0:j] @ z[0:j] */
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[i * nb + k] * sz[k];
                sT[i * nb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[j * nb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * nb; idx += nt) {        /* write panel back */
        int r = idx / nb, c = idx % nb;
        Bb[(size_t)(p0 + r) * n + (p0 + c)] = sP[idx];
    }
    for (int idx = tid; idx < nb * nb; idx += nt)         /* write T */
        Tg[(size_t)b * nb * nb + idx] = sT[idx];
    for (int i = tid; i < nb; i += nt)                    /* write tau */
        taug[(size_t)b * n + (p0 + i)] = stau[i];
}

void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb) {
    int batch = B.size(0), n = B.size(1);
    int m = n - p0;
    size_t shmem = ((size_t)m * nb + (size_t)nb * nb + 2 * nb + 256) * sizeof(float);
    cudaFuncSetAttribute(panel_geqrt_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_geqrt_kernel<<<batch, 256, shmem>>>(
        B.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), n, p0, nb);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
"""
CPP_SRC = "void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb);"

_K = load_inline(name="qr_hh", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                 functions=["panel_geqrt"], extra_cuda_cflags=["-O3"], verbose=False)

GEQRF_N = 1536          # n >= this: one-block-per-matrix starves occupancy -> torch.geqrf wins


def _nb_for(n):
    return max(8, min(64, 30000 // n))          # keep the FP32 panel m*nb <= ~120KB SMEM


def _hh_qr(A):
    b, n, _ = A.shape
    B = A.contiguous().clone()
    tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    p0 = 0
    while p0 < n:
        nb = min(_nb_for(n), n - p0)
        pe = p0 + nb
        T = torch.zeros(b, nb, nb, device=A.device, dtype=A.dtype)
        _K.panel_geqrt(B, T, tau, p0, nb)            # (1) panel: reflectors, R, tau, T
        if pe < n:                                    # (2) WY trailing update (BLAS-3)
            V = torch.tril(B[:, p0:, p0:pe], -1)
            V[:, :nb, :nb] += eye[:nb, :nb]
            C = B[:, p0:, pe:]
            VtC = torch.einsum('bmi,bmc->bic', V, C)
            TtVtC = torch.einsum('bki,bkc->bic', T, VtC)
            B[:, p0:, pe:] = C - torch.einsum('bmi,bic->bmc', V, TtVtC)
        p0 = pe
    return B, tau


def custom_kernel(data: input_t) -> output_t:
    A = data
    n = A.shape[-1]
    if n >= GEQRF_N:
        return torch.geqrf(A)
    return _hh_qr(A)
