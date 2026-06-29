"""Recursive/blocked-HH de-risk: the panel apply (~95% of panel work) is currently
in-SMEM warp-FMA. A blocked panel would do it as efficient GEMMs (cuBLAS now, BF16x9
tensor-core later). Question: does a batched GEMM beat warp-FMA at the panel-INTERNAL
apply sizes (smaller than between-panel), enough to justify leaving SMEM?

Compares, at b=640 m=512 and the sub-block sizes a recursive panel creates:
  warp-FMA WY-apply (our in-kernel style)  vs  batched at::matmul (cuBLAS)
Also FP32 vs the allow_tf32 path (cuBLAS tensor-core) as the upper bound.

  source /workspace/qr/env.sh && python householder/panel_apply_derisk.py
"""
import sys, pathlib
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
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
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"launch");
}
"""
K = load_inline(name="panel_apply_derisk", cpp_sources=["void btrail(torch::Tensor,torch::Tensor,torch::Tensor);"],
                cuda_sources=[CUDA_SRC], functions=["btrail"], extra_cuda_cflags=["-O3"], verbose=False)


def _t(fn, r=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / r


def cublas_apply(V, T, C):
    W = torch.matmul(V.transpose(1, 2), C); W = torch.matmul(T.transpose(1, 2), W); C.sub_(torch.matmul(V, W))


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("batched WY-apply at panel-internal sizes (b=640, m=512): warp-FMA vs cuBLAS")
    print(f"{'k':>3s}{'ncol':>6s} {'warpFMA':>9s} {'cuBLAS32':>9s} {'cuBLAS-tf32':>12s} {'FMA/cb32':>9s}")
    g = torch.Generator(device="cuda").manual_seed(0)
    for k in (8, 16, 32):
        for ncol in (16, 32):
            V = torch.randn(640, 512, k, device="cuda", generator=g).contiguous()
            T = torch.triu(torch.randn(640, k, k, device="cuda", generator=g)).contiguous()
            C0 = torch.randn(640, 512, ncol, device="cuda", generator=g).contiguous()
            C1 = C0.clone()
            torch.backends.cuda.matmul.allow_tf32 = False
            tf = _t(lambda: K.btrail(V, T, C0))
            tc = _t(lambda: cublas_apply(V, T, C1.clone() if False else C1))
            torch.backends.cuda.matmul.allow_tf32 = True
            ctf = _t(lambda: cublas_apply(V, T, C1))
            torch.backends.cuda.matmul.allow_tf32 = False
            print(f"{k:>3d}{ncol:>6d} {tf*1e3:>8.1f}u {tc*1e3:>8.1f}u {ctf*1e3:>11.1f}u {tf/tc:>8.1f}x")
