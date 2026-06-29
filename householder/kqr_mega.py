"""THE BIG KERNEL — fused blocked Householder QR, single launch, fully on-chip.

One CTA per matrix. Internal block-column loop (no per-block cuBLAS launches).
  panel  : geqr2 in SMEM (warp-parallel) -> V (reflectors), tau, R, and T-factor
  trailing: C -= V (T^T (V^T C))  done IN-KERNEL via tensor-core wmma (tf32), C in global
Output H (V below diag, R on/above) + tau == geqrf-compatible, no reconstruction.

Stage 1 (this file): correct fused skeleton, warp panel + in-kernel mma trailing.
Stage 2 (next): swap panel to recursive-WY mma (the panel speed win).

  source /workspace/qr/env.sh && python householder/kqr_mega.py
"""
import sys, pathlib, importlib.util
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check                          # noqa: E402

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <vector>
using namespace nvcuda;

__device__ __forceinline__ float warp_sum(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o); return v; }
__device__ __forceinline__ float warp_max(float v){ for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o)); return v; }

// tf32 wmma tile op:  acc(16x16) += A(16x16) * B(16x16), A,B in SMEM as float (tf32-rounded)
#define WM 16
#define WN 16
#define WK 8

/* Fused blocked Householder QR. One CTA per matrix, 256 threads.
   SMEM: sP = panel (m x LD), sT = T (kb x kb), staging for trailing.
   Trailing columns processed in panels of width TW. */
#define TWID 32
__global__ void qr_mega_kernel(float* __restrict__ Hm, float* __restrict__ taug, int n, int NB, int do_trail){
    extern __shared__ float smem[];
    int b=blockIdx.x, tid=threadIdx.x, nt=blockDim.x, warp=tid>>5, lane=tid&31, nwarp=nt>>5;
    float* Hb = Hm + (size_t)b*n*n;
    __shared__ float s_tau, s_denom;

    for(int p0=0; p0<n; p0+=NB){
        int kb = min(NB, n-p0), m = n-p0, LD = kb+1;
        int mpad=((m+15)/16)*16, kpad=((kb+15)/16)*16;
        float* sP = smem;                            // m x LD
        float* sT = sP + (size_t)m*LD;               // kb x kb
        float* sz = sT + (size_t)kb*kb;              // kb
        float* sstau = sz + kb;                      // kb
        float* sVf = sstau + kb;                     // m x kb    (materialized V, fp32 for tf32 wmma)
        float* sW  = sVf + (size_t)m*kb;             // kb x 64   (V^T C)
        float* sW2 = sW + (size_t)kb*64;             // kb x 64   (T^T W)
        float* sCacc = sW2 + (size_t)kb*64;          // nwarp x 256 (per-warp C tile scratch)
        (void)mpad; (void)kpad;

        // load panel Hb[p0:n, p0:p0+kb]
        for(int idx=tid; idx<m*kb; idx+=nt){ int r=idx/kb,c=idx%kb; sP[(size_t)r*LD+c]=Hb[(size_t)(p0+r)*n+(p0+c)]; }
        for(int idx=tid; idx<kb*kb; idx+=nt) sT[idx]=0.f;
        __syncthreads();

        // ---- panel geqr2 (warp-parallel, in SMEM) ----
        for(int j=0;j<kb;++j){
            if(warp==0){
                float amax=0.f; for(int r=j+1+lane;r<m;r+=32) amax=fmaxf(amax,fabsf(sP[(size_t)r*LD+j])); amax=warp_max(amax);
                float ssq=0.f; if(amax>0.f) for(int r=j+1+lane;r<m;r+=32){ float t=sP[(size_t)r*LD+j]/amax; ssq+=t*t; } ssq=warp_sum(ssq);
                float xnorm=(amax>0.f)?amax*sqrtf(ssq):0.f;
                if(lane==0){ float alpha=sP[(size_t)j*LD+j],beta,tj,den;
                    if(xnorm==0.f){beta=alpha;tj=0.f;den=1.f;} else {beta=-copysignf(hypotf(alpha,xnorm),alpha);tj=(beta-alpha)/beta;den=alpha-beta;}
                    sP[(size_t)j*LD+j]=beta; sstau[j]=tj; s_tau=tj; s_denom=den; }
                __syncwarp(); float den=s_denom; for(int r=j+1+lane;r<m;r+=32) sP[(size_t)r*LD+j]/=den;
            }
            __syncthreads(); float tj=s_tau;
            for(int c=j+1+warp;c<kb;c+=nwarp){ float p=0.f; for(int r=j+1+lane;r<m;r+=32) p+=sP[(size_t)r*LD+j]*sP[(size_t)r*LD+c];
                float w=(warp_sum(p)+sP[(size_t)j*LD+c])*tj; if(lane==0) sP[(size_t)j*LD+c]-=w; for(int r=j+1+lane;r<m;r+=32) sP[(size_t)r*LD+c]-=sP[(size_t)r*LD+j]*w; }
            __syncthreads();
        }
        // ---- build T (DLARFT, warp-per-col-i) ----
        for(int j=0;j<kb;++j){
            if(j>0){ for(int i=warp;i<j;i+=nwarp){ float d=0.f; for(int r=j+1+lane;r<m;r+=32) d+=sP[(size_t)r*LD+i]*sP[(size_t)r*LD+j]; d=warp_sum(d);
                if(lane==0) sz[i]=-sstau[j]*(d+sP[(size_t)j*LD+i]); }
                __syncthreads();
                for(int i=tid;i<j;i+=nt){ float a=0.f; for(int k=i;k<j;++k) a+=sT[(size_t)i*kb+k]*sz[k]; sT[(size_t)i*kb+j]=a; }
                __syncthreads(); }
            if(tid==0) sT[(size_t)j*kb+j]=sstau[j];
            __syncthreads();
        }
        // ---- write panel back: V (below diag, unit), R (on/above) ----
        for(int idx=tid; idx<m*kb; idx+=nt){ int r=idx/kb,c=idx%kb; Hb[(size_t)(p0+r)*n+(p0+c)]=sP[(size_t)r*LD+c]; }
        for(int i=tid;i<kb;i+=nt) taug[(size_t)b*n+(p0+i)]=sstau[i];
        __syncthreads();

        // ---- trailing WY update: C -= V (T^T (V^T C)),  TENSOR CORE (tf32 wmma), wide col-blocks ----
        // (board shapes: m,kb,ncol multiples of 16 with NB=32 -> no padding / no OOB)
        int pe=p0+kb, ncol=n-pe;
        if(do_trail && ncol>0){
            for(int idx=tid; idx<m*kb; idx+=nt){ int r=idx/kb,c=idx%kb;
                sVf[(size_t)r*kb+c] = (r==c)?1.f : ((r>c)?sP[(size_t)r*LD+c]:0.f); }
            __syncthreads();
            const int NCB = 64;                         // column block; W/W2 are kb x NCB
            for(int c0=0; c0<ncol; c0+=NCB){
                int ncb = min(NCB, ncol-c0), ntn = ncb/16, krt = kb/16;
                // W = V^T C  (kb x ncb): all (krt x ntn) tiles across warps, accumulate over m
                for(int t=warp; t<krt*ntn; t+=nwarp){ int mt=t/ntn, nj=t%ntn;
                    wmma::fragment<wmma::accumulator,16,16,8,float> acc; wmma::fill_fragment(acc,0.f);
                    for(int ks=0; ks<m/8; ++ks){
                        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> a;
                        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bb;
                        wmma::load_matrix_sync(a, sVf + mt*16 + (size_t)ks*8*kb, kb);
                        wmma::load_matrix_sync(bb, Hb + (size_t)(p0+ks*8)*n + (pe+c0+nj*16), n);
                        for(int i=0;i<a.num_elements;++i) a.x[i]=wmma::__float_to_tf32(a.x[i]);
                        for(int i=0;i<bb.num_elements;++i) bb.x[i]=wmma::__float_to_tf32(bb.x[i]);
                        wmma::mma_sync(acc,a,bb,acc);
                    }
                    wmma::store_matrix_sync(sW + (size_t)mt*16*NCB + nj*16, acc, NCB, wmma::mem_row_major);
                }
                __syncthreads();
                // W2 = T^T W  (kb x ncb)
                for(int idx=tid; idx<kb*ncb; idx+=nt){ int i=idx/ncb, c=idx%ncb;
                    float a=0.f; for(int k=0;k<=i;++k) a+=sT[(size_t)k*kb+i]*sW[(size_t)k*NCB+c];
                    sW2[(size_t)i*NCB+c]=a; }
                __syncthreads();
                // C -= V W2  (m x ncb): all (m/16 x ntn) tiles across warps, accumulate over kb
                for(int t=warp; t<(m/16)*ntn; t+=nwarp){ int rt=t/ntn, nj=t%ntn;
                    wmma::fragment<wmma::accumulator,16,16,8,float> acc; wmma::fill_fragment(acc,0.f);
                    for(int ks=0; ks<kb/8; ++ks){
                        wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bb;
                        wmma::load_matrix_sync(a, sVf + (size_t)rt*16*kb + ks*8, kb);
                        wmma::load_matrix_sync(bb, sW2 + (size_t)ks*8*NCB + nj*16, NCB);
                        for(int i=0;i<a.num_elements;++i) a.x[i]=wmma::__float_to_tf32(a.x[i]);
                        for(int i=0;i<bb.num_elements;++i) bb.x[i]=wmma::__float_to_tf32(bb.x[i]);
                        wmma::mma_sync(acc,a,bb,acc);
                    }
                    wmma::store_matrix_sync(sCacc + (size_t)(warp)*16*16, acc, 16, wmma::mem_row_major);
                    __syncwarp();
                    for(int e=lane; e<256; e+=32){ int rr=rt*16+e/16, cc=c0+nj*16+e%16;
                        Hb[(size_t)(p0+rr)*n+(pe+cc)] -= sCacc[(size_t)warp*256 + e]; }
                    __syncwarp();
                }
                __syncthreads();
            }
        }
    }
}

std::vector<torch::Tensor> qr_mega(torch::Tensor A, int64_t NB, int64_t do_trail){
    auto H = A.contiguous().clone();
    int b=H.size(0), n=H.size(1);
    auto tau = torch::zeros({b,n}, H.options());
    int kb=NB; size_t LD=kb+1;
    size_t floats = (size_t)n*LD + (size_t)kb*kb + kb + kb + (size_t)n*kb + (size_t)kb*64 + (size_t)kb*64 + 8*256;
    size_t shmem = floats*sizeof(float) + 256;
    cudaFuncSetAttribute(qr_mega_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem);
    qr_mega_kernel<<<b,256,shmem>>>(H.data_ptr<float>(), tau.data_ptr<float>(), n, (int)NB, (int)do_trail);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"qr_mega launch");
    return {H, tau};
}
"""
K = load_inline(name="kqr_mega", cpp_sources=["std::vector<torch::Tensor> qr_mega(torch::Tensor A, int64_t NB, int64_t do_trail);"],
                cuda_sources=[CUDA_SRC], functions=["qr_mega"], extra_cuda_cflags=["-O3"], verbose=False)


def custom_kernel(A, NB=32, do_trail=1):
    return K.qr_mega(A.contiguous(), NB, do_trail)


def _t(fn, A, r=20):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn(A)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / r


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("== kqr_mega: TENSOR-CORE trailing — correctness + speed (n512 b640, baseline ~12ms) ==")
    B, N = 640, 512
    worst = 0.0
    for case in ["dense", "mixed", "rankdef", "clustered"]:
        A = make_batch(B, N, 2, case, seed=0).cuda().contiguous()
        H, tau = custom_kernel(A, 32, 1)
        fr, og, ft, ot, ps = check(A, H, tau)
        worst = max(worst, fr/ft, og/ot)
        print(f"  n{N} {case:10s} margin {max(fr/ft,og/ot):.4f}  {'OK' if ps.all() else f'FAIL({int(ps.sum())}/{B})'}")
    A = make_batch(B, N, 2, "dense", seed=0).cuda().contiguous()
    t = _t(lambda x: custom_kernel(x, 32, 1), A)
    print(f"  worst margin {worst:.4f}   TIME {t*1e3:.1f}us  (baseline fused-cuBLAS ~12000us)")
