"""WMMA tensor-core Gram for the cluster KAN kernel -- de-risk + measure in isolation.

The scalar cluster Gram (kqr_cluster_kan.py) reads remote DSMEM with stride w in its
inner loop -- the Blackwell tuning guide flags non-unit-stride DSMEM as pathological and
says to stage into LOCAL shared memory first. Combined with the tensor-core thesis (the
metric GEMM A^T A is where the arithmetic should concentrate), the fix is:

  per 16x16 output tile of M[:,Jr], for each k-tile:
    * stage the remote A tile (matrix_a) and own A tile (matrix_b) FP32 -> 3 bf16 limbs
      into LOCAL per-warp shared staging (coalesced read, then unit-stride WMMA loads),
    * accumulate bf16x6 emulated-FP32 via 6 wmma::mma_sync,
  store the FP32 accumulator tile to M.

col_major load of a row-major staged tile yields its transpose => Aclo^T A gives M=A^T A.
Validate M == A^T A and time vs the scalar cluster Gram. Run on B200: python dev/kqr_cluster_wmma.py
"""
import torch
from torch.utils.cpp_extension import load_inline
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch

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

__device__ inline void f2b3(float x, __nv_bfloat16& h, __nv_bfloat16& m, __nv_bfloat16& l) {
    h = __float2bfloat16_rn(x);            float r1 = x  - __bfloat162float(h);
    m = __float2bfloat16_rn(r1);           float r2 = r1 - __bfloat162float(m);
    l = __float2bfloat16_rn(r2);
}

/* WMMA bf16x6 Gram: M[:,Jr] = A^T A[:,Jr].  Aloc = own column panel (n x w), Apan[s] =
   block s's panel (DSMEM). stage = per-warp scratch: 6 bf16 16x16 tiles (Ah,Am,Al,Bh,Bm,Bl). */
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
            for (int e = lane; e < 256; e += 32) {           /* stage matrix_a tile (remote) */
                int rr = e >> 4, cc = e & 15;
                f2b3(Asrc[(size_t)(kt * 16 + rr) * w + (loff + cc)], sAh[e], sAm[e], sAl[e]);
            }
            for (int e = lane; e < 256; e += 32) {           /* stage matrix_b tile (own) */
                int rr = e >> 4, cc = e & 15;
                f2b3(Aloc[(size_t)(kt * 16 + rr) * w + (ct * 16 + cc)], sBh[e], sBm[e], sBl[e]);
            }
            __syncwarp();
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ah, am, al;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bh, bm, bl;
            wmma::load_matrix_sync(ah, sAh, 16); wmma::load_matrix_sync(bh, sBh, 16);
            wmma::load_matrix_sync(am, sAm, 16); wmma::load_matrix_sync(bm, sBm, 16);
            wmma::mma_sync(cf, ah, bh, cf);                  /* h*h */
            wmma::mma_sync(cf, ah, bm, cf);                  /* h*m */
            wmma::mma_sync(cf, am, bh, cf);                  /* m*h  -> bf16x3 */
            if (order >= 2) {
                wmma::load_matrix_sync(al, sAl, 16); wmma::load_matrix_sync(bl, sBl, 16);
                wmma::mma_sync(cf, ah, bl, cf);              /* h*l */
                wmma::mma_sync(cf, al, bh, cf);              /* l*h */
                wmma::mma_sync(cf, am, bm, cf);              /* m*m -> bf16x6 */
            }
            __syncwarp();
        }
        wmma::store_matrix_sync(&Mloc[(size_t)(rt * 16) * w + ct * 16], cf, w, wmma::mem_row_major);
    }
}

__global__ void gram_wmma_kernel(const float* __restrict__ Ain, float* __restrict__ Mout,
                                 int n, int w, int order) {
    extern __shared__ float smem[];
    float* Aloc = smem; float* Mloc = Aloc + n * w;
    __nv_bfloat16* stage = (__nv_bfloat16*)(Mloc + n * w);
    cg::cluster_group cl = cg::this_cluster();
    int r = cl.block_rank(), C = cl.num_blocks();
    int bM = blockIdx.x / C, tid = threadIdx.x, nt = blockDim.x;
    const float* Ab = Ain + (size_t)bM * n * n;
    for (int idx = tid; idx < n * w; idx += nt) {
        int i = idx / w, jl = idx % w; Aloc[idx] = Ab[(size_t)i * n + (r * w + jl)];
    }
    cl.sync();
    float* Apan[16];
    for (int s = 0; s < C; ++s) Apan[s] = cl.map_shared_rank(Aloc, s);
    gram_wmma_cluster(Apan, Aloc, Mloc, stage, n, w, order, tid, nt);
    cl.sync();
    float* Mb = Mout + (size_t)bM * n * n;
    for (int idx = tid; idx < n * w; idx += nt) {
        int i = idx / w, jl = idx % w; Mb[(size_t)i * n + (r * w + jl)] = Mloc[idx];
    }
    cl.sync();
}

torch::Tensor cluster_gram_wmma(torch::Tensor A, int C, int order) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat32 && A.dim() == 3);
    A = A.contiguous(); int batch = A.size(0), n = A.size(1), w = n / C;
    TORCH_CHECK(n % C == 0 && w % 16 == 0 && C <= 16, "need n%C==0,(n/C)%16==0,C<=16");
    auto M = torch::empty_like(A);
    int nwarp = 256 / 32;
    size_t smem = (size_t)(2 * n * w) * sizeof(float)
                + (size_t)(nwarp * 6 * 256) * sizeof(__nv_bfloat16);
    cudaFuncSetAttribute(gram_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (C > 8) cudaFuncSetAttribute(gram_wmma_kernel,
                                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(batch * C, 1, 1); cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = C; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&cfg, gram_wmma_kernel, A.data_ptr<float>(),
                                       M.data_ptr<float>(), n, w, order);
    TORCH_CHECK(e == cudaSuccess, "launch: ", cudaGetErrorString(e));
    e = cudaDeviceSynchronize(); TORCH_CHECK(e == cudaSuccess, "sync: ", cudaGetErrorString(e));
    return M;
}
"""
CPP_SRC = "torch::Tensor cluster_gram_wmma(torch::Tensor A, int C, int order);"


def main():
    import time
    mod = load_inline(name="kqr_cluster_wmma", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                      functions=["cluster_gram_wmma"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_100,code=sm_100",
                                         "--expt-relaxed-constexpr"], verbose=True)

    print("WMMA tensor-core cluster Gram: correctness (M = A^T A)")
    for (b, n, C) in [(8, 256, 8), (8, 352, 11), (8, 512, 16)]:
        A = make_batch(b, n, 2, "dense", seed=0)
        ref = A.mT @ A
        for order in (1, 2):
            M = mod.cluster_gram_wmma(A, C, order)
            rel = (M - ref).abs().max() / ref.abs().max()
            print(f"  n={n:4d} C={C:2d} bf16x{3 if order==1 else 6}:  rel={rel:.2e}")

    print("\nTiming b640 n512 C16 (vs scalar Gram's ~307 ms/iter):")
    A = make_batch(640, 512, 2, "dense", seed=0)
    for order in (1, 2):
        mod.cluster_gram_wmma(A, 16, order)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): mod.cluster_gram_wmma(A, 16, order)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1e3
        print(f"  bf16x{3 if order==1 else 6}:  {ms:.1f} ms/iter")


if __name__ == "__main__":
    main()
