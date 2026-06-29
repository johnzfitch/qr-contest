#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200
"""Blocked WY-Householder QR (qr_v2) — warp-parallel FP32 panel (kqr_panel_v2, Phase 1).

Drop-in replacement of the scalar panel_geqrt with the warp-per-column / lane-over-rows
leaf (kqr_panel_v2). nb=32 (clamped to fit SMEM). Trailing update unchanged (fp32 WY).
Checkpoint of the first all-correct major speedup: n512 32.5->~17ms, n1024 ~35->~22ms.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* One CTA (256 threads = 8 warps) per matrix. Factor panel B[p0:n, p0:pe]
   (m=n-p0 rows, nb cols) by warp-parallel Householder, then DLARFT T. */
__global__ void panel_geqrt_v2_kernel(float* __restrict__ B, float* __restrict__ Tg,
                                      float* __restrict__ taug, int n, int p0, int nb) {
    extern __shared__ float smem[];
    int m = n - p0;
    int LD = nb + 1;                       /* padded stride: kills 32-way conflict */
    float* sP   = smem;                    /* m*LD  panel (row-major, padded)      */
    float* sT   = sP + (size_t)m * LD;     /* nb*nb T                              */
    float* stau = sT + (size_t)nb * nb;    /* nb    tau                            */
    float* sz   = stau + nb;               /* nb    DLARFT workspace               */
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
    float* Bb = B + (size_t)b * n * n;
    __shared__ float s_tau, s_denom;

    for (int idx = tid; idx < m * nb; idx += nt) {          /* load panel (padded) */
        int r = idx / nb, c = idx % nb;
        sP[(size_t)r * LD + c] = Bb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < nb * nb; idx += nt) sT[idx] = 0.f;
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (warp == 0) {                                   /* --- reflector j (path A) --- */
            float amax = 0.f;                              /* max-scaled ssq norm of tail  */
            for (int r = j + 1 + lane; r < m; r += 32)
                amax = fmaxf(amax, fabsf(sP[(size_t)r * LD + j]));
            amax = warp_max(amax);
            float ssq = 0.f;
            if (amax > 0.f)
                for (int r = j + 1 + lane; r < m; r += 32) {
                    float t = sP[(size_t)r * LD + j] / amax; ssq += t * t;
                }
            ssq = warp_sum(ssq);
            float xnorm = (amax > 0.f) ? amax * sqrtf(ssq) : 0.f;
            if (lane == 0) {
                float alpha = sP[(size_t)j * LD + j];
                float beta, tauj, denom;
                if (xnorm == 0.f) { beta = alpha; tauj = 0.f; denom = 1.f; }   /* identity */
                else {
                    beta  = -copysignf(hypotf(alpha, xnorm), alpha);
                    tauj  = (beta - alpha) / beta;          /* never 1/tau */
                    denom = alpha - beta;
                }
                sP[(size_t)j * LD + j] = beta; stau[j] = tauj; s_tau = tauj; s_denom = denom;
            }
            __syncwarp();
            float denom = s_denom;                          /* warp 0 scales tail: v=tail/denom */
            for (int r = j + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + j] /= denom;
        }
        __syncthreads();                                    /* publish v + tau */

        float tauj = s_tau;
        for (int c = j + 1 + warp; c < nb; c += nwarp) {    /* warp per trailing column */
            float partial = 0.f;
            for (int r = j + 1 + lane; r < m; r += 32)
                partial += sP[(size_t)r * LD + j] * sP[(size_t)r * LD + c];
            float w = (warp_sum(partial) + sP[(size_t)j * LD + c]) * tauj;   /* v[j]=1 term */
            if (lane == 0) sP[(size_t)j * LD + c] -= w;
            for (int r = j + 1 + lane; r < m; r += 32)
                sP[(size_t)r * LD + c] -= sP[(size_t)r * LD + j] * w;
        }
        __syncthreads();                                    /* update complete */
    }

    /* ---- DLARFT: compact-WY T (nb x nb upper-tri), forward columnwise ---- */
    for (int j = 0; j < nb; ++j) {
        if (j > 0) {
            for (int i = tid; i < j; i += nt) {            /* z[i] = -tau_j * (V[:,i].V[:,j]) */
                float d = sP[(size_t)j * LD + i];
                for (int r = j + 1; r < m; ++r) d += sP[(size_t)r * LD + i] * sP[(size_t)r * LD + j];
                sz[i] = -stau[j] * d;
            }
            __syncthreads();
            for (int i = tid; i < j; i += nt) {            /* T[0:j,j] = T[0:j,0:j] @ z[0:j] */
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[(size_t)i * nb + k] * sz[k];
                sT[(size_t)i * nb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[(size_t)j * nb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * nb; idx += nt) {          /* write panel back */
        int r = idx / nb, c = idx % nb;
        Bb[(size_t)(p0 + r) * n + (p0 + c)] = sP[(size_t)r * LD + c];
    }
    for (int idx = tid; idx < nb * nb; idx += nt) Tg[(size_t)b * nb * nb + idx] = sT[idx];
    for (int i = tid; i < nb; i += nt) taug[(size_t)b * n + (p0 + i)] = stau[i];
}

void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb) {
    int batch = B.size(0), n = B.size(1);
    int m = n - p0, LD = nb + 1;
    size_t shmem = ((size_t)m * LD + (size_t)nb * nb + 2 * nb + 32) * sizeof(float);
    cudaFuncSetAttribute(panel_geqrt_v2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_geqrt_v2_kernel<<<batch, 256, shmem>>>(
        B.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), n, p0, nb);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
"""
CPP_SRC = "void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb);"

_K = load_inline(name="kqr_panel_v2_sub", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                 functions=["panel_geqrt"], extra_cuda_cflags=["-O3"], verbose=False)

GEQRF_N = 1536
SMEM_FLOATS = 56 * 1024

def _nb_for(n):
    nb = min(32, n)
    while nb > 8 and n * (nb + 1) + nb * nb + 2 * nb + 32 > SMEM_FLOATS:
        nb -= 8
    return nb

def _hh_qr(A):
    b, n, _ = A.shape
    B = A.contiguous().clone()
    tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    p0 = 0
    while p0 < n:
        nb = min(_nb_for(n), n - p0); pe = p0 + nb
        T = torch.zeros(b, nb, nb, device=A.device, dtype=A.dtype)
        _K.panel_geqrt(B, T, tau, p0, nb)
        if pe < n:
            V = torch.tril(B[:, p0:, p0:pe], -1); V[:, :nb, :nb] += eye[:nb, :nb]
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
