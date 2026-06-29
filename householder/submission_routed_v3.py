#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = False

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <vector>
#include <algorithm>

#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

__global__ void panel_geqrt_v2_kernel(float* __restrict__ Hm, float* __restrict__ Tg,
                                      float* __restrict__ taug, float* __restrict__ Vout,
                                      int n, int p0, int kb, int ldt, int ldv) {
    extern __shared__ float smem[];
    int m = n - p0, LD = kb + 1;
    float* sP   = smem;
    float* sT   = sP + (size_t)m * LD;
    float* stau = sT + (size_t)kb * kb;
    float* sz   = stau + kb;
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
    float* Hb = Hm + (size_t)b * n * n;
    __shared__ float s_tau, s_denom;

    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, c = idx % kb;
        sP[(size_t)r * LD + c] = Hb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < kb * kb; idx += nt) sT[idx] = 0.f;
    __syncthreads();

    for (int j = 0; j < kb; ++j) {
        if (warp == 0) {
            float amax = 0.f;
            for (int r = j + 1 + lane; r < m; r += 32) amax = fmaxf(amax, fabsf(sP[(size_t)r * LD + j]));
            amax = warp_max(amax);
            float ssq = 0.f;
            if (amax > 0.f) for (int r = j + 1 + lane; r < m; r += 32) { float t = sP[(size_t)r * LD + j] / amax; ssq += t * t; }
            ssq = warp_sum(ssq);
            float xnorm = (amax > 0.f) ? amax * sqrtf(ssq) : 0.f;
            if (lane == 0) {
                float alpha = sP[(size_t)j * LD + j], beta, tauj, denom;
                if (xnorm == 0.f) { beta = alpha; tauj = 0.f; denom = 1.f; }
                else { beta = -copysignf(hypotf(alpha, xnorm), alpha); tauj = (beta - alpha) / beta; denom = alpha - beta; }
                sP[(size_t)j * LD + j] = beta; stau[j] = tauj; s_tau = tauj; s_denom = denom;
            }
            __syncwarp();
            float denom = s_denom;
            for (int r = j + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + j] /= denom;
        }
        __syncthreads();
        float tauj = s_tau;
        for (int c = j + 1 + warp; c < kb; c += nwarp) {
            float partial = 0.f;
            for (int r = j + 1 + lane; r < m; r += 32) partial += sP[(size_t)r * LD + j] * sP[(size_t)r * LD + c];
            float w = (warp_sum(partial) + sP[(size_t)j * LD + c]) * tauj;
            if (lane == 0) sP[(size_t)j * LD + c] -= w;
            for (int r = j + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + c] -= sP[(size_t)r * LD + j] * w;
        }
        __syncthreads();
    }

    for (int j = 0; j < kb; ++j) {                 /* DLARFT T -- warp-per-i, lanes over rows */
        if (j > 0) {
            for (int i = warp; i < j; i += nwarp) {
                float d = 0.f;
                for (int r = j + 1 + lane; r < m; r += 32) d += sP[(size_t)r * LD + i] * sP[(size_t)r * LD + j];
                d = warp_sum(d);
                if (lane == 0) sz[i] = -stau[j] * (d + sP[(size_t)j * LD + i]);
            }
            __syncthreads();
            for (int i = tid; i < j; i += nt) {
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[(size_t)i * kb + k] * sz[k];
                sT[(size_t)i * kb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[(size_t)j * kb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, c = idx % kb;
        float val = sP[(size_t)r * LD + c];
        Hb[(size_t)(p0 + r) * n + (p0 + c)] = val;
        float vv = (r == c) ? 1.f : ((r > c) ? val : 0.f);
        Vout[(size_t)b * n * ldv + (size_t)(p0 + r) * ldv + c] = vv;
    }
    for (int i = tid; i < kb; i += nt)
        for (int j = 0; j < kb; ++j) Tg[(size_t)b * ldt * ldt + (size_t)i * ldt + j] = sT[(size_t)i * kb + j];
    for (int i = tid; i < kb; i += nt) taug[(size_t)b * n + (p0 + i)] = stau[i];
}

__global__ void panel_wavefront_v3_kernel(float* __restrict__ Hm, float* __restrict__ Tg,
                                          float* __restrict__ taug, float* __restrict__ Vout,
                                          int n, int p0, int kb, int ldt, int ldv) {
    extern __shared__ float smem[];
    int m = n - p0, LD = kb + 1;
    float* sP   = smem;
    float* sT   = sP + (size_t)m * LD;
    float* stau = sT + (size_t)kb * kb;
    float* sz   = stau + kb;
    int*   ready = (int*)(sz + kb);
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int warp = tid >> 5, lane = tid & 31;
    float* Hb = Hm + (size_t)b * n * n;

    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, c = idx % kb;
        sP[(size_t)r * LD + c] = Hb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < kb * kb; idx += nt) sT[idx] = 0.f;
    for (int idx = tid; idx < kb;      idx += nt) ready[idx] = 0;
    __syncthreads();

    int c = warp;
    if (c < kb) {
        for (int k = 0; k < c; ++k) {
            cuda::atomic_ref<int, cuda::thread_scope_block> fr(ready[k]);
            while (fr.load(cuda::memory_order_acquire) == 0) { __nanosleep(32); }
            float tauk = stau[k];
            float partial = 0.f;
            for (int r = k + 1 + lane; r < m; r += 32)
                partial += sP[(size_t)r * LD + k] * sP[(size_t)r * LD + c];
            float w = (warp_sum(partial) + sP[(size_t)k * LD + c]) * tauk;
            if (lane == 0) sP[(size_t)k * LD + c] -= w;
            for (int r = k + 1 + lane; r < m; r += 32)
                sP[(size_t)r * LD + c] -= sP[(size_t)r * LD + k] * w;
            __syncwarp(FULL);
        }
        float amax = 0.f;
        for (int r = c + 1 + lane; r < m; r += 32) amax = fmaxf(amax, fabsf(sP[(size_t)r * LD + c]));
        amax = warp_max(amax);
        float ssq = 0.f;
        if (amax > 0.f) for (int r = c + 1 + lane; r < m; r += 32) { float t = sP[(size_t)r * LD + c] / amax; ssq += t * t; }
        ssq = warp_sum(ssq);
        float xnorm = (amax > 0.f) ? amax * sqrtf(ssq) : 0.f;
        float denom;
        if (lane == 0) {
            float alpha = sP[(size_t)c * LD + c], beta, tauc;
            if (xnorm == 0.f) { beta = alpha; tauc = 0.f; denom = 1.f; }
            else { beta = -copysignf(hypotf(alpha, xnorm), alpha); tauc = (beta - alpha) / beta; denom = alpha - beta; }
            sP[(size_t)c * LD + c] = beta; stau[c] = tauc;
        }
        denom = __shfl_sync(FULL, denom, 0);
        for (int r = c + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + c] /= denom;
        __syncwarp(FULL);
        __threadfence_block();
        __syncwarp(FULL);
        if (lane == 0)
            cuda::atomic_ref<int, cuda::thread_scope_block>(ready[c]).store(1, cuda::memory_order_release);
    }
    __syncthreads();

    for (int j = 0; j < kb; ++j) {
        if (j > 0) {
            for (int i = tid; i < j; i += nt) {
                float d = sP[(size_t)j * LD + i];
                for (int r = j + 1; r < m; ++r) d += sP[(size_t)r * LD + i] * sP[(size_t)r * LD + j];
                sz[i] = -stau[j] * d;
            }
            __syncthreads();
            for (int i = tid; i < j; i += nt) {
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[(size_t)i * kb + k] * sz[k];
                sT[(size_t)i * kb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[(size_t)j * kb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, cc = idx % kb;
        float val = sP[(size_t)r * LD + cc];
        Hb[(size_t)(p0 + r) * n + (p0 + cc)] = val;
        float vv = (r == cc) ? 1.f : ((r > cc) ? val : 0.f);
        Vout[(size_t)b * n * ldv + (size_t)(p0 + r) * ldv + cc] = vv;
    }
    for (int i = tid; i < kb; i += nt)
        for (int j = 0; j < kb; ++j) Tg[(size_t)b * ldt * ldt + (size_t)i * ldt + j] = sT[(size_t)i * kb + j];
    for (int i = tid; i < kb; i += nt) taug[(size_t)b * n + (p0 + i)] = stau[i];
}

static void launch_panel(torch::Tensor H, torch::Tensor T, torch::Tensor tau,
                         torch::Tensor V, int p0, int kb, int mode) {
    int batch = H.size(0), n = H.size(1), m = n - p0, LD = kb + 1;
    int ldt = T.size(2), ldv = V.size(2);
    if (mode == 1) {
        TORCH_CHECK(kb <= 32, "wavefront is one-warp-per-column: kb must be <= 32");
        size_t shmem = ((size_t)m * LD + (size_t)kb * kb + 3 * kb + 32) * sizeof(float);
        int block = kb * 32;
        cudaFuncSetAttribute(panel_wavefront_v3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        panel_wavefront_v3_kernel<<<batch, block, shmem>>>(
            H.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), V.data_ptr<float>(),
            n, p0, kb, ldt, ldv);
    } else {
        size_t shmem = ((size_t)m * LD + (size_t)kb * kb + 2 * kb + 32) * sizeof(float);
        cudaFuncSetAttribute(panel_geqrt_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        panel_geqrt_v2_kernel<<<batch, 256, shmem>>>(
            H.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), V.data_ptr<float>(),
            n, p0, kb, ldt, ldv);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "panel launch failed");
}

std::vector<torch::Tensor> qr_blocked(torch::Tensor A, int64_t NB, int64_t mode) {
    auto H = A.contiguous().clone();
    int64_t batch = H.size(0), n = H.size(1);
    auto opt = H.options();
    auto tau = torch::zeros({batch, n}, opt);
    auto T = torch::empty({batch, NB, NB}, opt);
    auto V = torch::zeros({batch, n, NB}, opt);
    for (int64_t p0 = 0; p0 < n; ) {
        int64_t kb = std::min(NB, n - p0), pe = p0 + kb;
        launch_panel(H, T, tau, V, (int)p0, (int)kb, (int)mode);
        if (pe < n) {
            auto Vt = V.narrow(1, p0, n - p0).narrow(2, 0, kb);
            auto C  = H.narrow(1, p0, n - p0).narrow(2, pe, n - pe);
            auto W  = at::matmul(Vt.transpose(1, 2), C);
            auto Tk = T.narrow(1, 0, kb).narrow(2, 0, kb);
            W = at::matmul(Tk.transpose(1, 2), W);
            C.sub_(at::matmul(Vt, W));
        }
        p0 = pe;
    }
    return {H, tau};
}
"""
CPP_SRC = "std::vector<torch::Tensor> qr_blocked(torch::Tensor A, int64_t NB, int64_t mode);"

_K = load_inline(name="kqr_routed_sub_v3", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                 functions=["qr_blocked"], extra_cuda_cflags=["-O3"], verbose=False)

GEQRF_N = 4096          # only n>=4096 (b2) routes to geqrf
SMEM_FLOATS = 58368     # 228KB/4 (matches kqr_fused_v2)


def _fit_nb_fused(n, NB=32):
    nb = min(NB, n)
    while nb > 8 and n * (nb + 1) + nb * nb + 2 * nb + 32 > SMEM_FLOATS:
        nb -= 8
    return nb


def _fit_nb_wave(n, NB=32):
    nb = min(NB, n)
    while nb > 8 and n * (nb + 1) + nb * nb + 3 * nb + 32 > SMEM_FLOATS:
        nb -= 8
    return min(nb, 32)


def custom_kernel(data: input_t) -> output_t:
    A = data
    n = A.shape[-1]
    if n >= GEQRF_N:
        return torch.geqrf(A)
    Ac = A.contiguous()
    if n >= 512:                     # DLARFT-parallel fused wins 512/1024/2048
        H, tau = _K.qr_blocked(Ac, _fit_nb_fused(n), 0)
    else:                            # wavefront wins tiny m (176/352)
        H, tau = _K.qr_blocked(Ac, _fit_nb_wave(n), 1)
    return H, tau
