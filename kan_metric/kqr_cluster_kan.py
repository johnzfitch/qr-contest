"""Phase-2 CLUSTER KAN kernel (n >= 256) -- the fused Iwasawa pipeline across a cluster.

ONE thread-block CLUSTER per matrix. The n x n working set is distributed across the
cluster's C blocks in a COLUMN-PANEL layout: block rank r owns columns [r*w, (r+1)*w)
of every logical matrix, with X[i,j] stored in block (j/w) at local index i*w + (j%w).
Any block reads another block's panel via cg::cluster.map_shared_rank() (DSMEM). The
mapped pointers depend only on the SMEM base, not the contents, so we map each panel
ONCE after load and reuse the pointers as the buffers evolve through the pipeline.

Same Iwasawa-KAN pipeline as single-block dev/kqr_fused.py, cooperative across a cluster:
  A -> M=A^T A (Gram) -> R=chol(M) -> Q=A R^-1 -> CQR2 -> modified-LU(Q) -> factor-closure
       R_out = S.(Q^T A) -> write H = tril(L,-1) + triu(R_out), tau = 2/||v||^2
All stages blocked at panel width w => each costs O(C) cluster barriers, not O(n).

Two entry points share the device stages:
  cluster_kan(A,C) -> (Q,M)   : debug, runs to STAGE (1 Gram /2 +chol /3 +trsm /4 +CQR2)
  cluster_qr(A,C)  -> (H,tau) : the full fused KAN factorization (geqrf contract)

w = n/C multiple of 16.  n=352->C=11(w=32);  n=512->C=16(w=32, non-portable).
Run on the Runpod B200:  python dev/kqr_cluster_kan.py
"""
import torch
from torch.utils.cpp_extension import load_inline
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check

torch.backends.cuda.matmul.allow_tf32 = False

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <vector>
namespace cg = cooperative_groups;
using namespace nvcuda;

#ifndef STAGE
#define STAGE 4
#endif

/* QSTAGE: stage-profiling early-exit for cluster_qr_kernel (b640n512 breakdown).
   1 = stop after CQR2 (2x WMMA-Gram+chol+trsm); 2 = +modified-LU; 3 = +closure (full). */
#ifndef QSTAGE
#define QSTAGE 3
#endif

/* WMMA bf16x6 tensor-core Gram (validated in kqr_cluster_wmma.py). M[:,Jr]=A^T A[:,Jr].
   Stages remote+own A tiles FP32->3 bf16 limbs into per-warp LOCAL shared `stage`
   (avoids strided DSMEM per the Blackwell tuning guide), then bf16x{3,6} via WMMA. */
__device__ inline void f2b3(float x, __nv_bfloat16& h, __nv_bfloat16& m, __nv_bfloat16& l) {
    h = __float2bfloat16_rn(x);  float r1 = x  - __bfloat162float(h);
    m = __float2bfloat16_rn(r1); float r2 = r1 - __bfloat162float(m);
    l = __float2bfloat16_rn(r2);
}
__device__ void gram_wmma_cluster(float** Apan, float* Aloc, float* Mloc,
                                  __nv_bfloat16* stage, int n, int w, int order,
                                  int tid, int nt) {
    int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
    int RT = n / 16, CT = w / 16, KT = n / 16;
    __nv_bfloat16* sAh = stage + (size_t)warp * 6 * 256;
    __nv_bfloat16* sAm = sAh + 256; __nv_bfloat16* sAl = sAm + 256;
    __nv_bfloat16* sBh = sAl + 256; __nv_bfloat16* sBm = sBh + 256; __nv_bfloat16* sBl = sBm + 256;
    for (int ot = warp; ot < RT * CT; ot += nwarp) {
        int rt = ot / CT, ct = ot % CT;
        int gcol0 = rt * 16, owner = gcol0 / w, loff = gcol0 - owner * w;
        float* Asrc = Apan[owner];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.0f);
        for (int kt = 0; kt < KT; ++kt) {
            for (int e = lane; e < 256; e += 32) {
                int rr = e >> 4, cc = e & 15;
                f2b3(Asrc[(size_t)(kt * 16 + rr) * w + (loff + cc)], sAh[e], sAm[e], sAl[e]);
            }
            for (int e = lane; e < 256; e += 32) {
                int rr = e >> 4, cc = e & 15;
                f2b3(Aloc[(size_t)(kt * 16 + rr) * w + (ct * 16 + cc)], sBh[e], sBm[e], sBl[e]);
            }
            __syncwarp();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ah, am, al;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bh, bm, bl;
            wmma::load_matrix_sync(ah, sAh, 16); wmma::load_matrix_sync(bh, sBh, 16);
            wmma::load_matrix_sync(am, sAm, 16); wmma::load_matrix_sync(bm, sBm, 16);
            wmma::mma_sync(cf, ah, bh, cf);
            wmma::mma_sync(cf, ah, bm, cf);
            wmma::mma_sync(cf, am, bh, cf);
            if (order >= 2) {
                wmma::load_matrix_sync(al, sAl, 16); wmma::load_matrix_sync(bl, sBl, 16);
                wmma::mma_sync(cf, ah, bl, cf);
                wmma::mma_sync(cf, al, bh, cf);
                wmma::mma_sync(cf, am, bm, cf);
            }
            __syncwarp();
        }
        wmma::store_matrix_sync(&Mloc[(size_t)(rt * 16) * w + ct * 16], cf, w, wmma::mem_row_major);
    }
}

/* ---- distributed stage device fns (column-panel layout, panel width w) ---- *
 * Aloc/Mloc : this block's n x w panel of A and M (row-major: [i*w + localcol]).
 * Apan/Mpan : Apan[s] -> block s's Aloc (DSMEM); valid for the kernel's lifetime.
 * Logical X[i,j] = Xpan[j/w][i*w + (j%w)].  r = this block's rank, C = #blocks.    */

/* Gram: Mloc = src^T src[:,Jr], src panels in span[] (src own panel = sloc). */
__device__ void gram_cluster(float** span, float* sloc, float* Mloc,
                             int n, int w, int tid, int nt) {
    for (int out = tid; out < n * w; out += nt) {
        int i = out / w, jl = out % w;
        const float* Si = span[i / w];
        int il = i % w;
        float s = 0.f;
        for (int k = 0; k < n; ++k) s += Si[k * w + il] * sloc[k * w + jl];
        Mloc[out] = s;
    }
}

/* in-block upper Cholesky of a w x w block D (row stride w): D <- chol so D^T D = D_in. */
__device__ void chol_blk(float* D, int w, int tid, int nt) {
    for (int j = 0; j < w; ++j) {
        if (tid == 0) {
            float s = D[j * w + j];
            for (int k = 0; k < j; ++k) { float rr = D[k * w + j]; s -= rr * rr; }
            D[j * w + j] = (s > 1e-30f) ? sqrtf(s) : 1e-15f;
        }
        __syncthreads();
        float rjj = D[j * w + j];
        for (int i = j + 1 + tid; i < w; i += nt) {
            float s = D[j * w + i];
            for (int k = 0; k < j; ++k) s -= D[k * w + j] * D[k * w + i];
            D[j * w + i] = s / rjj;
        }
        __syncthreads();
    }
}

/* Distributed right-looking blocked Cholesky: M(upper) -> R(upper), M = R^T R. */
__device__ void chol_cluster(cg::cluster_group& cl, float** Mpan, float* Mloc,
                             int n, int w, int r, int C, int tid, int nt) {
    for (int p = 0; p < C; ++p) {
        if (r == p) chol_blk(Mloc + (size_t)p * w * w, w, tid, nt);
        cl.sync();
        if (r > p) {                       /* R[Jp,Jr] = R[Jp,Jp]^-T M[Jp,Jr] */
            float* Rpp = Mpan[p];
            for (int a = 0; a < w; ++a) {
                float raa = Rpp[(size_t)(p * w + a) * w + a];
                for (int c = tid; c < w; c += nt) {
                    float s = Mloc[(size_t)(p * w + a) * w + c];
                    for (int e = 0; e < a; ++e)
                        s -= Rpp[(size_t)(p * w + e) * w + a] * Mloc[(size_t)(p * w + e) * w + c];
                    Mloc[(size_t)(p * w + a) * w + c] = s / raa;
                }
                __syncthreads();
            }
        }
        cl.sync();
        if (r > p) {                       /* trailing: M[i>p,Jr] -= R[Jp,i]^T R[Jp,Jr] */
            int i0 = (p + 1) * w;
            for (int out = tid; out < (n - i0) * w; out += nt) {
                int i = i0 + out / w, c = out % w;
                float* Ri = Mpan[i / w]; int il = i % w;
                float s = 0.f;
                for (int kl = 0; kl < w; ++kl)
                    s += Ri[(size_t)(p * w + kl) * w + il] * Mloc[(size_t)(p * w + kl) * w + c];
                Mloc[(size_t)i * w + c] -= s;
            }
        }
        cl.sync();
    }
}

/* Distributed right-looking right-solve: A <- A R^-1 (R upper, in Mloc). */
__device__ void trsm_cluster(cg::cluster_group& cl, float** Apan, float* Aloc, float* Mloc,
                             int n, int w, int r, int C, int tid, int nt) {
    for (int p = 0; p < C; ++p) {
        if (r == p) {                      /* Q[:,Jp] = A[:,Jp] R[Jp,Jp]^-1 */
            for (int c = 0; c < w; ++c) {
                float rcc = Mloc[(size_t)(p * w + c) * w + c];
                for (int i = tid; i < n; i += nt) {
                    float s = Aloc[(size_t)i * w + c];
                    for (int k = 0; k < c; ++k)
                        s -= Aloc[(size_t)i * w + k] * Mloc[(size_t)(p * w + k) * w + c];
                    Aloc[(size_t)i * w + c] = s / rcc;
                }
                __syncthreads();
            }
        }
        cl.sync();
        if (r > p) {                       /* A[:,Jr] -= Q[:,Jp] R[Jp,Jr] */
            float* Qp = Apan[p];
            for (int out = tid; out < n * w; out += nt) {
                int i = out / w, c = out % w;
                float s = 0.f;
                for (int kl = 0; kl < w; ++kl)
                    s += Qp[(size_t)i * w + kl] * Mloc[(size_t)(p * w + kl) * w + c];
                Aloc[(size_t)i * w + c] -= s;
            }
        }
        cl.sync();
    }
}

/* Distributed right-looking blocked modified-LU of (B - S) = L U, in place on Bloc
   (= a COPY of the orthonormal Q). Emits L (below diag), per-column signs sgnloc,
   tau = 2/||v||^2. Mirrors the validated single-block / blocked_modlu. */
__device__ void modlu_cluster(cg::cluster_group& cl, float** Bpan, float* Bloc, float* sgnloc,
                              float* tau_out, int n, int w, int r, int C, int bM,
                              int tid, int nt) {
    for (int p = 0; p < C; ++p) {
        if (r == p) {                      /* factor tall panel p (cols Jp, rows >= p*w) */
            for (int il = 0; il < w; ++il) {
                int i = p * w + il;
                if (tid == 0) {
                    float d = Bloc[(size_t)i * w + il];
                    float s = (d > 0.f) ? -1.f : 1.f;
                    Bloc[(size_t)i * w + il] = d - s;   /* pivot, |piv| in [1,2] */
                    sgnloc[il] = s;
                }
                __syncthreads();
                float piv = Bloc[(size_t)i * w + il];
                for (int j = i + 1 + tid; j < n; j += nt) Bloc[(size_t)j * w + il] /= piv;
                __syncthreads();
                int m = n - (i + 1), wd = w - (il + 1);
                for (int out = tid; out < m * wd; out += nt) {
                    int j = i + 1 + out / wd, kl = il + 1 + out % wd;
                    Bloc[(size_t)j * w + kl] -= Bloc[(size_t)j * w + il] * Bloc[(size_t)i * w + kl];
                }
                __syncthreads();
            }
        }
        cl.sync();
        if (r > p) {                       /* U12: B[Jp,Jr] <- L11^-1 B[Jp,Jr] (unit lower) */
            float* L11 = Bpan[p];
            for (int a = 1; a < w; ++a) {
                for (int c = tid; c < w; c += nt) {
                    float s = Bloc[(size_t)(p * w + a) * w + c];
                    for (int b = 0; b < a; ++b)
                        s -= L11[(size_t)(p * w + a) * w + b] * Bloc[(size_t)(p * w + b) * w + c];
                    Bloc[(size_t)(p * w + a) * w + c] = s;
                }
                __syncthreads();
            }
        }
        cl.sync();
        if (r > p) {                       /* trailing: B[i>p,Jr] -= L21 U12 */
            float* L21 = Bpan[p];
            int i0 = (p + 1) * w;
            for (int out = tid; out < (n - i0) * w; out += nt) {
                int i = i0 + out / w, c = out % w;
                float s = 0.f;
                for (int kl = 0; kl < w; ++kl)
                    s += L21[(size_t)i * w + kl] * Bloc[(size_t)(p * w + kl) * w + c];
                Bloc[(size_t)i * w + c] -= s;
            }
        }
        cl.sync();
    }
    for (int il = tid; il < w; il += nt) {            /* tau for own columns */
        int i = r * w + il;
        float ss = 1.f;
        for (int j = i + 1; j < n; ++j) { float v = Bloc[(size_t)j * w + il]; ss += v * v; }
        tau_out[(size_t)bM * n + i] = 2.f / ss;
    }
}

/* ===================== debug kernel (stages 1..4) ===================== */
__global__ void cluster_kan_kernel(const float* __restrict__ Ain,
                                   float* __restrict__ Qout, float* __restrict__ Mout,
                                   int n, int w) {
    extern __shared__ float smem[];
    float* Aloc = smem; float* Mloc = Aloc + n * w;
    cg::cluster_group cl = cg::this_cluster();
    int r = cl.block_rank(), C = cl.num_blocks();
    int bM = blockIdx.x / C, tid = threadIdx.x, nt = blockDim.x;
    const float* Ab = Ain + (size_t)bM * n * n;
    for (int idx = tid; idx < n * w; idx += nt) {
        int i = idx / w, jl = idx % w; Aloc[idx] = Ab[(size_t)i * n + (r * w + jl)];
    }
    cl.sync();
    float* Apan[16]; float* Mpan[16];
    for (int s = 0; s < C; ++s) { Apan[s] = cl.map_shared_rank(Aloc, s);
                                  Mpan[s] = cl.map_shared_rank(Mloc, s); }
    gram_cluster(Apan, Aloc, Mloc, n, w, tid, nt); cl.sync();
#if STAGE >= 2
    chol_cluster(cl, Mpan, Mloc, n, w, r, C, tid, nt);
#endif
#if STAGE >= 3
    trsm_cluster(cl, Apan, Aloc, Mloc, n, w, r, C, tid, nt);
#endif
#if STAGE >= 4
    gram_cluster(Apan, Aloc, Mloc, n, w, tid, nt); cl.sync();
    chol_cluster(cl, Mpan, Mloc, n, w, r, C, tid, nt);
    trsm_cluster(cl, Apan, Aloc, Mloc, n, w, r, C, tid, nt);
#endif
    float* Qb = Qout + (size_t)bM * n * n; float* Mb = Mout + (size_t)bM * n * n;
    for (int idx = tid; idx < n * w; idx += nt) {
        int i = idx / w, jl = idx % w; int j = r * w + jl;
        Qb[(size_t)i * n + j] = Aloc[idx]; Mb[(size_t)i * n + j] = Mloc[idx];
    }
    cl.sync();
}

/* ===================== full fused KAN kernel -> (H, tau) ===================== */
__global__ void cluster_qr_kernel(const float* __restrict__ Ain, float* __restrict__ Hout,
                                  float* __restrict__ tauout, int n, int w) {
    extern __shared__ float smem[];
    float* Aloc = smem;             /* A -> Q1 -> Q2 (kept live)  (n*w) */
    float* Mloc = Aloc + n * w;     /* M -> R, then Q2-copy -> L  (n*w) */
    float* sgnloc = Mloc + n * w;   /* per-column signs           (w)   */
    __nv_bfloat16* stage = (__nv_bfloat16*)(sgnloc + w);  /* WMMA Gram staging */
    cg::cluster_group cl = cg::this_cluster();
    int r = cl.block_rank(), C = cl.num_blocks();
    int bM = blockIdx.x / C, tid = threadIdx.x, nt = blockDim.x;
    const float* Ab = Ain + (size_t)bM * n * n;
    for (int idx = tid; idx < n * w; idx += nt) {
        int i = idx / w, jl = idx % w; Aloc[idx] = Ab[(size_t)i * n + (r * w + jl)];
    }
    cl.sync();
    float* Apan[16]; float* Mpan[16]; float* Span[16];
    for (int s = 0; s < C; ++s) { Apan[s] = cl.map_shared_rank(Aloc, s);
                                  Mpan[s] = cl.map_shared_rank(Mloc, s);
                                  Span[s] = cl.map_shared_rank(sgnloc, s); }
    /* CQR2 (WMMA tensor-core Gram, bf16x6) */
    gram_wmma_cluster(Apan, Aloc, Mloc, stage, n, w, 2, tid, nt); cl.sync();
    chol_cluster(cl, Mpan, Mloc, n, w, r, C, tid, nt);
    trsm_cluster(cl, Apan, Aloc, Mloc, n, w, r, C, tid, nt);
    gram_wmma_cluster(Apan, Aloc, Mloc, stage, n, w, 2, tid, nt); cl.sync();
    chol_cluster(cl, Mpan, Mloc, n, w, r, C, tid, nt);
    trsm_cluster(cl, Apan, Aloc, Mloc, n, w, r, C, tid, nt);
#if QSTAGE <= 1            /* profile: stop after CQR2 (write Q2 to H, timing only) */
    { float* Hb = Hout + (size_t)bM * n * n;
      for (int idx = tid; idx < n * w; idx += nt) { int i = idx / w, jl = idx % w;
          Hb[(size_t)i * n + (r * w + jl)] = Aloc[idx]; }
      cl.sync(); return; }
#endif
    /* Aloc = Q2 (kept). modified-LU on a COPY in Mloc. */
    for (int idx = tid; idx < n * w; idx += nt) Mloc[idx] = Aloc[idx];
    cl.sync();
    modlu_cluster(cl, Mpan, Mloc, sgnloc, tauout, n, w, r, C, bM, tid, nt);
    cl.sync();
#if QSTAGE <= 2            /* profile: stop after modified-LU (write L to H, timing only) */
    { float* Hb = Hout + (size_t)bM * n * n;
      for (int idx = tid; idx < n * w; idx += nt) { int i = idx / w, jl = idx % w;
          Hb[(size_t)i * n + (r * w + jl)] = Mloc[idx]; }
      cl.sync(); return; }
#endif
    /* factor closure: H below diag = L (Mloc); on/above = sgn[i]*(Q2^T A)[i,j] */
    float* Hb = Hout + (size_t)bM * n * n;
    for (int out = tid; out < n * w; out += nt) {
        int i = out / w, jl = out % w; int j = r * w + jl;
        if (i > j) {
            Hb[(size_t)i * n + j] = Mloc[out];
        } else {
            const float* Qi = Apan[i / w]; int il = i % w;
            float acc = 0.f;
            for (int k = 0; k < n; ++k) acc += Qi[(size_t)k * w + il] * Ab[(size_t)k * n + j];
            Hb[(size_t)i * n + j] = Span[i / w][i % w] * acc;
        }
    }
    cl.sync();
}

static void set_attrs(const void* fn, size_t smem, int C) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (C > 8) cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}
static cudaLaunchConfig_t make_cfg(int batch, int C, size_t smem, cudaLaunchAttribute* attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(batch * C, 1, 1); cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smem;
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1; return cfg;
}

std::vector<torch::Tensor> cluster_kan(torch::Tensor A, int C) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32 && A.dim() == 3);
    A = A.contiguous(); int batch = A.size(0), n = A.size(1), w = n / C;
    TORCH_CHECK(n % C == 0 && w % 16 == 0 && C <= 16, "need n%C==0, (n/C)%16==0, C<=16");
    auto Q = torch::empty_like(A), M = torch::empty_like(A);
    size_t smem = (size_t)(2 * n * w) * sizeof(float);
    set_attrs((const void*)cluster_kan_kernel, smem, C);
    cudaLaunchAttribute attr[1]; auto cfg = make_cfg(batch, C, smem, attr);
    cudaError_t e = cudaLaunchKernelEx(&cfg, cluster_kan_kernel,
                                       A.data_ptr<float>(), Q.data_ptr<float>(),
                                       M.data_ptr<float>(), n, w);
    TORCH_CHECK(e == cudaSuccess, "launch: ", cudaGetErrorString(e));
    e = cudaDeviceSynchronize(); TORCH_CHECK(e == cudaSuccess, "sync: ", cudaGetErrorString(e));
    return {Q, M};
}

std::vector<torch::Tensor> cluster_qr(torch::Tensor A, int C) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32 && A.dim() == 3);
    A = A.contiguous(); int batch = A.size(0), n = A.size(1), w = n / C;
    TORCH_CHECK(n % C == 0 && w % 16 == 0 && C <= 16, "need n%C==0, (n/C)%16==0, C<=16");
    auto H = torch::empty_like(A);
    auto tau = torch::empty({batch, n}, A.options());
    size_t smem = (size_t)(2 * n * w + w) * sizeof(float)
                + (size_t)((256 / 32) * 6 * 256) * sizeof(__nv_bfloat16);  /* WMMA staging */
    set_attrs((const void*)cluster_qr_kernel, smem, C);
    cudaLaunchAttribute attr[1]; auto cfg = make_cfg(batch, C, smem, attr);
    cudaError_t e = cudaLaunchKernelEx(&cfg, cluster_qr_kernel,
                                       A.data_ptr<float>(), H.data_ptr<float>(),
                                       tau.data_ptr<float>(), n, w);
    TORCH_CHECK(e == cudaSuccess, "launch: ", cudaGetErrorString(e));
    e = cudaDeviceSynchronize(); TORCH_CHECK(e == cudaSuccess, "sync: ", cudaGetErrorString(e));
    return {H, tau};
}
"""
CPP_SRC = ("std::vector<torch::Tensor> cluster_kan(torch::Tensor A, int C);\n"
           "std::vector<torch::Tensor> cluster_qr(torch::Tensor A, int C);")


def _build(stage=4, qstage=3):
    return load_inline(
        name=f"kqr_cluster_kan_s{stage}q{qstage}", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
        functions=["cluster_kan", "cluster_qr"],
        extra_cuda_cflags=["-O3", f"-DSTAGE={stage}", f"-DQSTAGE={qstage}",
                           "-gencode=arch=compute_100,code=sm_100",
                           "--expt-relaxed-constexpr"], verbose=True)


def profile_stages(b=640, n=512, C=16, cond=2):
    """Stage breakdown at the critical shape: CQR2 -> +modLU -> +closure (cumulative ms)."""
    A = make_batch(b, n, cond, "dense", seed=0)
    labels = {1: "CQR2 (2x gram+chol+trsm)", 2: "+ modified-LU", 3: "+ closure (full)"}
    print(f"\nStage breakdown b{b} n{n} C{C} (cumulative, then per-stage delta):")
    cum = {}
    for q in (1, 2, 3):
        mod = _build(4, q)
        cum[q] = _bench(lambda: mod.cluster_qr(A, C))
        print(f"  QSTAGE{q} {labels[q]:28s}: {cum[q]:7.1f} ms")
    print(f"  --- deltas ---  CQR2={cum[1]:.1f}  modLU={cum[2]-cum[1]:.1f}  "
          f"closure={cum[3]-cum[2]:.1f}  (ms)")


SHAPES = [(8, 256, 8), (8, 352, 11), (8, 512, 16)]
CFOR = {352: 11, 512: 16}     # cluster size per n


def _bench(fn, iters=3):
    import time
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    mod = _build(4, 3)   # full pipeline (cached name reused by profile_stages' QSTAGE3)

    print("Correctness sanity (full kernel unchanged by QSTAGE instrumentation):")
    for (b, n, C) in [(8, 512, 16)]:
        A = make_batch(b, n, 2, "dense", seed=0)
        H, tau = mod.cluster_qr(A, C)
        fr, og, ft, ot, ps = check(A, H, tau)
        print(f"  n={n} C={C}:  factor={fr:.2e}/{ft:.2e}  ortho={og:.2e}/{ot:.2e}  "
              f"pass={int(ps.sum())}/{len(ps)}  {'OK' if ps.all() else 'FAIL'}")

    print("\nBaseline timing (b640 n512 C16 dense c2):")
    A = make_batch(640, 512, 2, "dense", seed=0)
    print(f"  full cluster_qr:  {_bench(lambda: mod.cluster_qr(A, 16)):.1f} ms/iter")

    profile_stages(b=640, n=512, C=16, cond=2)


if __name__ == "__main__":
    main()
