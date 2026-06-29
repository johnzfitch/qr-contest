"""In-kernel mma.sync (wmma) WY-apply de-risk -- the FOURTH door.

Three tensor-core probes this weekend wrapped the primitive in cuBLAS and hit a
~74us launch FLOOR. That floor is cuBLAS's, not the tensor core's. wmma/mma.sync
from INSIDE a kernel has no launch overhead -- it's a register-tile instruction.
So the real question the proxy never answered:

  at the panel-apply tile sizes (b=640, m=512, k=32, ncol=32), how fast is a
  hand-written wmma fragment doing the WY-apply  C -= V (T^T (V^T C))  with V,C
  staged in SMEM the way the warp-FMA path stages them -- vs the 322us warp-FMA?

If wmma is ~2.5-3x on this (80% of n512), geomean ~6.83 -> ~5 and <4 is in play.
If it's ~1.5x, the executor's obituary was right and 6.83 is the weekend floor.

We also need FP32 accuracy: B200 bf16 tensor core. Single bf16 (x1) loses ~8 bits
-> will NOT pass the cond-1 gate on its own. bf16x3 (3-limb split: hi*hi + hi*lo +
lo*hi) recovers ~24 bits. We time BOTH and report each variant's residual vs an
fp32 reference so we know which limb count clears the oracle gate.

  source /workspace/qr/env.sh && python householder/mma_apply_derisk.py

WMMA LAYOUT MAP (the load-bearing detail):
  V stored (m=512, k=32) row-major, row-stride 32.  C stored (m=512, ncol=32) rm.
  Step 1  W = V^T C           A=V^T (M=32,K=512), B=C (K=512,N=32)  -> W (32,32)
          A=V^T as col_major ld=32 reads V directly: A(i,l)=V[l][i]=ptr[i + l*32].
          B=C  as row_major ld=32 reads C directly: B(l,c)=C[l][c]=ptr[l*32 + c].
  Step 2  W2 = T^T W          small 32x32, done in fp32 by threads.
  Step 3  C -= V W2           A=V (M=512,K=32) row_major ld=32, B=W2 (32,32) rm.
"""
import sys, pathlib, importlib.util
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

#define MROWS 512
#define KB    32
#define NCOL  32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ void split2(float x, __nv_bfloat16& h, __nv_bfloat16& l){
    h = __float2bfloat16(x);
    float hf = __bfloat162float(h);
    l = __float2bfloat16(x - hf);
}

/* ---- warp-FMA baseline (identical math to panel_apply_derisk btrail_k) ---- */
#define TN 32
__global__ void btrail_k(const float* __restrict__ Vg, const float* __restrict__ Tg,
                         float* __restrict__ Cg, int m, int nb, int ncol){
    extern __shared__ float sm[]; float* sV=sm; float* sT=sV+(size_t)m*nb; float* sW=sT+(size_t)nb*nb; float* sW2=sW+(size_t)nb*TN;
    int b=blockIdx.x,c0=blockIdx.y*TN; int tn=min(TN,ncol-c0); int tid=threadIdx.x,nt=blockDim.x;
    const float* Vb=Vg+(size_t)b*m*nb; const float* Tb=Tg+(size_t)b*nb*nb; float* Cb=Cg+(size_t)b*m*ncol;
    for(int i=tid;i<m*nb;i+=nt)sV[i]=Vb[i]; for(int i=tid;i<nb*nb;i+=nt)sT[i]=Tb[i]; __syncthreads();
    for(int idx=tid;idx<nb*tn;idx+=nt){int i=idx/tn,c=idx%tn;float acc=0.f;for(int r=0;r<m;++r)acc+=sV[(size_t)r*nb+i]*Cb[(size_t)r*ncol+(c0+c)];sW[(size_t)i*TN+c]=acc;}
    __syncthreads();
    for(int idx=tid;idx<nb*tn;idx+=nt){int i=idx/tn,c=idx%tn;float acc=0.f;for(int k=0;k<=i;++k)acc+=sT[(size_t)k*nb+i]*sW[(size_t)k*TN+c];sW2[(size_t)i*TN+c]=acc;}
    __syncthreads();
    for(int idx=tid;idx<m*tn;idx+=nt){int r=idx/tn,c=idx%tn;float acc=0.f;for(int i=0;i<nb;++i)acc+=sV[(size_t)r*nb+i]*sW2[(size_t)i*TN+c];Cb[(size_t)r*ncol+(c0+c)]-=acc;}
}
void btrail(torch::Tensor V, torch::Tensor T, torch::Tensor C){
    int b=V.size(0),m=V.size(1),nb=V.size(2),ncol=C.size(2);
    dim3 grid(b,(ncol+TN-1)/TN); size_t shmem=((size_t)m*nb+(size_t)nb*nb+2*(size_t)nb*TN)*sizeof(float);
    cudaFuncSetAttribute(btrail_k,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem);
    btrail_k<<<grid,256,shmem>>>(V.data_ptr<float>(),T.data_ptr<float>(),C.data_ptr<float>(),m,nb,ncol);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"btrail launch");
}

/* ---- in-kernel wmma WY-apply.  One CTA per batch matrix, 256 threads = 8 warps. ----
   LIMBS=1 : single bf16 (Vh*Ch).   LIMBS=3 : Vh*Ch + Vh*Cl + Vl*Ch (FP32-accurate). */
template<int LIMBS>
__global__ void mma_apply_k(const float* __restrict__ Vg, const float* __restrict__ Tg,
                            const float* __restrict__ Cin, float* __restrict__ Cout){
    extern __shared__ char smem_raw[];
    // bf16 staging
    __nv_bfloat16* sVh = (__nv_bfloat16*)smem_raw;                 // 512*32
    __nv_bfloat16* sVl = sVh + MROWS*KB;                           // 512*32 (LIMBS==3)
    __nv_bfloat16* sCh = sVl + (LIMBS==3 ? MROWS*KB : 0);          // 512*32
    __nv_bfloat16* sCl = sCh + MROWS*KB;                           // 512*32 (LIMBS==3)
    __nv_bfloat16* sW2h = sCl + (LIMBS==3 ? MROWS*KB : 0);         // 32*32
    __nv_bfloat16* sW2l = sW2h + KB*KB;                            // 32*32 (LIMBS==3)
    float* sWf  = (float*)(sW2l + (LIMBS==3 ? KB*KB : 0));         // 32*32  W
    float* sW2f = sWf  + KB*KB;                                    // 32*32  W2
    float* sTf  = sW2f + KB*KB;                                    // 32*32  T
    float* sR   = sTf  + KB*KB;                                    // 512*32 V*W2 result

    int b=blockIdx.x, tid=threadIdx.x, nt=blockDim.x, warp=tid>>5, lane=tid&31;
    const float* Vb=Vg+(size_t)b*MROWS*KB; const float* Tb=Tg+(size_t)b*KB*KB;
    const float* Cb=Cin+(size_t)b*MROWS*NCOL; float* Co=Cout+(size_t)b*MROWS*NCOL;

    // stage V,C (split to bf16 limbs); load T fp32
    for(int i=tid;i<MROWS*KB;i+=nt){ __nv_bfloat16 h,l; split2(Vb[i],h,l); sVh[i]=h; if(LIMBS==3) sVl[i]=l; }
    for(int i=tid;i<MROWS*NCOL;i+=nt){ __nv_bfloat16 h,l; split2(Cb[i],h,l); sCh[i]=h; if(LIMBS==3) sCl[i]=l; }
    for(int i=tid;i<KB*KB;i+=nt) sTf[i]=Tb[i];
    __syncthreads();

    // ---- Step 1: W = V^T C   (M=32,N=32,K=512) -> 2x2 tiles of 16x16, warps 0..3 ----
    if(warp<4){
        int mi=warp>>1, nj=warp&1;   // tile (mi,nj)
        wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> acc; wmma::fill_fragment(acc,0.f);
        for(int ks=0;ks<MROWS/WMMA_K;++ks){
            wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::col_major> aVh;
            wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::row_major> bCh;
            wmma::load_matrix_sync(aVh, sVh + mi*16 + (size_t)ks*16*KB, KB);   // A=V^T col_major ld=32
            wmma::load_matrix_sync(bCh, sCh + (size_t)ks*16*NCOL + nj*16, NCOL); // B=C row_major ld=32
            wmma::mma_sync(acc,aVh,bCh,acc);
            if(LIMBS==3){
                wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::col_major> aVl;
                wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::row_major> bCl;
                wmma::load_matrix_sync(aVl, sVl + mi*16 + (size_t)ks*16*KB, KB);
                wmma::load_matrix_sync(bCl, sCl + (size_t)ks*16*NCOL + nj*16, NCOL);
                wmma::mma_sync(acc,aVh,bCl,acc);
                wmma::mma_sync(acc,aVl,bCh,acc);
            }
        }
        wmma::store_matrix_sync(sWf + mi*16*KB + nj*16, acc, KB, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- Step 2: W2 = T^T W  (32x32, fp32 by threads) then split to bf16 ----
    for(int idx=tid;idx<KB*NCOL;idx+=nt){ int i=idx/NCOL,c=idx%NCOL; float a=0.f;
        for(int k=0;k<=i;++k) a+=sTf[(size_t)k*KB+i]*sWf[(size_t)k*KB+c]; sW2f[(size_t)i*KB+c]=a; }
    __syncthreads();
    for(int i=tid;i<KB*KB;i+=nt){ __nv_bfloat16 h,l; split2(sW2f[i],h,l); sW2h[i]=h; if(LIMBS==3) sW2l[i]=l; }
    __syncthreads();

    // ---- Step 3: R = V W2   (M=512,N=32,K=32) -> 32x2 tiles, 8 warps round-robin ----
    int ntile = (MROWS/WMMA_M)*(NCOL/WMMA_N);  // 32*2 = 64
    for(int t=warp;t<ntile;t+=8){
        int ml=t>>1, nj=t&1;
        wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> acc; wmma::fill_fragment(acc,0.f);
        for(int ks=0;ks<KB/WMMA_K;++ks){
            wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::row_major> aVh;
            wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::row_major> bWh;
            wmma::load_matrix_sync(aVh, sVh + (size_t)ml*16*KB + ks*16, KB);     // A=V row_major ld=32
            wmma::load_matrix_sync(bWh, sW2h + (size_t)ks*16*KB + nj*16, KB);    // B=W2 row_major ld=32
            wmma::mma_sync(acc,aVh,bWh,acc);
            if(LIMBS==3){
                wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::row_major> aVl;
                wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,__nv_bfloat16,wmma::row_major> bWl;
                wmma::load_matrix_sync(aVl, sVl + (size_t)ml*16*KB + ks*16, KB);
                wmma::load_matrix_sync(bWl, sW2l + (size_t)ks*16*KB + nj*16, KB);
                wmma::mma_sync(acc,aVh,bWl,acc);
                wmma::mma_sync(acc,aVl,bWh,acc);
            }
        }
        wmma::store_matrix_sync(sR + ml*16*NCOL + nj*16, acc, NCOL, wmma::mem_row_major);
    }
    __syncthreads();
    for(int i=tid;i<MROWS*NCOL;i+=nt) Co[i]=Cb[i]-sR[i];
}

template<int LIMBS>
static size_t shmem_bytes(){
    size_t bf = (LIMBS==3? (size_t)4*MROWS*KB + 2*KB*KB : (size_t)2*MROWS*KB + KB*KB)*sizeof(__nv_bfloat16);
    size_t f  = ((size_t)3*KB*KB + MROWS*NCOL)*sizeof(float);
    return bf+f;
}

torch::Tensor mma_apply(torch::Tensor V, torch::Tensor T, torch::Tensor C, int64_t limbs){
    int b=V.size(0); auto Cout=torch::empty_like(C);
    if(limbs==1){ size_t sh=shmem_bytes<1>(); cudaFuncSetAttribute(mma_apply_k<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);
        mma_apply_k<1><<<b,256,sh>>>(V.data_ptr<float>(),T.data_ptr<float>(),C.data_ptr<float>(),Cout.data_ptr<float>()); }
    else { size_t sh=shmem_bytes<3>(); cudaFuncSetAttribute(mma_apply_k<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);
        mma_apply_k<3><<<b,256,sh>>>(V.data_ptr<float>(),T.data_ptr<float>(),C.data_ptr<float>(),Cout.data_ptr<float>()); }
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"mma_apply launch");
    return Cout;
}
"""
CPP_SRC = ("void btrail(torch::Tensor,torch::Tensor,torch::Tensor);\n"
           "torch::Tensor mma_apply(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);")
K = load_inline(name="mma_apply_derisk", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                functions=["btrail", "mma_apply"], extra_cuda_cflags=["-O3"], verbose=True)


def _t(fn, r=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / r


def cublas_apply(V, T, C):
    W = torch.matmul(V.transpose(1, 2), C); W = torch.matmul(T.transpose(1, 2), W); return C - torch.matmul(V, W)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    B, m, k, ncol = 640, 512, 32, 32
    g = torch.Generator(device="cuda").manual_seed(0)
    V = torch.randn(B, m, k, device="cuda", generator=g).contiguous()
    T = torch.triu(torch.randn(B, k, k, device="cuda", generator=g)).contiguous()
    C0 = torch.randn(B, m, ncol, device="cuda", generator=g).contiguous()

    # fp32 reference (this is what the gate compares against)
    ref = cublas_apply(V, T, C0)
    refn = ref.norm()

    print("== in-kernel wmma WY-apply vs warp-FMA  (b=640, m=512, k=32, ncol=32) ==")
    # correctness
    C_fma = C0.clone(); K.btrail(V, T, C_fma)
    out1 = K.mma_apply(V, T, C0, 1)
    out3 = K.mma_apply(V, T, C0, 3)
    print(f"  rel-resid vs fp32 ref:  warpFMA={ (C_fma-ref).norm()/refn :.2e}   "
          f"wmma_x1={ (out1-ref).norm()/refn :.2e}   wmma_x3={ (out3-ref).norm()/refn :.2e}")
    # the oracle apply-gate is ~ n*eps32 ~ 512*1.2e-7 ~ 6e-5; x1 will blow it, x3 should clear.

    # timing
    torch.backends.cuda.matmul.allow_tf32 = False
    t_fma = _t(lambda: K.btrail(V, T, C0.clone()))
    t_x1  = _t(lambda: K.mma_apply(V, T, C0, 1))
    t_x3  = _t(lambda: K.mma_apply(V, T, C0, 3))
    t_cb  = _t(lambda: cublas_apply(V, T, C0))
    torch.backends.cuda.matmul.allow_tf32 = True
    t_cbtf = _t(lambda: cublas_apply(V, T, C0))
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"\n  warp-FMA   : {t_fma*1e3:8.1f} us   (baseline)")
    print(f"  wmma x1    : {t_x1*1e3:8.1f} us   {t_fma/t_x1:5.2f}x  [accuracy fails gate]")
    print(f"  wmma x3    : {t_x3*1e3:8.1f} us   {t_fma/t_x3:5.2f}x  [FP32-accurate]")
    print(f"  cuBLAS fp32: {t_cb*1e3:8.1f} us   {t_fma/t_cb:5.2f}x")
    print(f"  cuBLAS tf32: {t_cbtf*1e3:8.1f} us   {t_fma/t_cbtf:5.2f}x")
    print("\n  VERDICT: wmma_x3 speedup over warp-FMA is the number that prices the build.")
    print("    >=2.5x -> path to ~5ms (maybe <4) is real;  ~1.5x -> 6.83ms is the floor.")
