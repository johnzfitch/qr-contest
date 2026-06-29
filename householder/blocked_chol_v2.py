"""Blocked Cholesky v2: warp-per-column triangular inverse (kills the super-cubic
inverse of v1) -> large nb viable -> few host iterations + big tensor-core trailing.

Diagonal kernel: right-looking chol in SMEM (fp32) + WARP-per-column forward-sub for
L11^-1 (lanes parallelize the inner sum, warp owns the sequential i-chain).
Panel L21 = M21 @ L11^-T and trailing M22 -= L21 L21^T via cuBLAS tf32 (tensor core).

  source /workspace/qr/env.sh && python householder/blocked_chol_v2.py
"""
import sys, pathlib
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch                                  # noqa: E402

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
__device__ __forceinline__ float warp_sum(float v){ for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffffu,v,o); return v; }

__global__ void diag_chol_inv2_k(const float* __restrict__ Mblk, float* __restrict__ Lout,
                                 float* __restrict__ Linvout, int nb){
    extern __shared__ float sm[];
    float* sL = sm; float* sX = sL + (size_t)nb*nb;
    int bid=blockIdx.x, tid=threadIdx.x, nt=blockDim.x, warp=tid>>5, lane=tid&31, nwarp=nt>>5;
    const float* Mb = Mblk + (size_t)bid*nb*nb;
    for(int i=tid;i<nb*nb;i+=nt) sL[i]=Mb[i];
    __syncthreads();
    // right-looking Cholesky (lower)
    for(int j=0;j<nb;++j){
        if(tid==0) sL[(size_t)j*nb+j]=sqrtf(sL[(size_t)j*nb+j]);
        __syncthreads();
        float ljj=sL[(size_t)j*nb+j];
        for(int i=j+1+tid;i<nb;i+=nt) sL[(size_t)i*nb+j]/=ljj;
        __syncthreads();
        for(int idx=tid; idx<nb*nb; idx+=nt){ int i=idx/nb, c=idx%nb;
            if(c>j && i>=c) sL[(size_t)i*nb+c]-=sL[(size_t)i*nb+j]*sL[(size_t)c*nb+j]; }
        __syncthreads();
    }
    for(int idx=tid; idx<nb*nb; idx+=nt){ int i=idx/nb,c=idx%nb; if(c>i) sL[idx]=0.f; }
    __syncthreads();
    // L^-1: WARP per column, lanes parallelize inner sum, warp owns sequential i-chain
    for(int col=warp; col<nb; col+=nwarp){
        for(int i=lane;i<col;i+=32) sX[(size_t)i*nb+col]=0.f;
        __syncwarp();
        for(int i=col;i<nb;++i){
            float partial=0.f;
            for(int k=col+lane;k<i;k+=32) partial += sL[(size_t)i*nb+k]*sX[(size_t)k*nb+col];
            float s = warp_sum(partial);
            if(lane==0){ float rhs=(i==col)?1.f:0.f; sX[(size_t)i*nb+col]=(rhs - s)/sL[(size_t)i*nb+i]; }
            __syncwarp();
        }
    }
    __syncthreads();
    for(int i=tid;i<nb*nb;i+=nt){ Lout[(size_t)bid*nb*nb+i]=sL[i]; Linvout[(size_t)bid*nb*nb+i]=sX[i]; }
}

std::vector<torch::Tensor> diag_chol_inv2(torch::Tensor D){
    int b=D.size(0), nb=D.size(1);
    auto L=torch::empty_like(D), Linv=torch::empty_like(D);
    size_t shmem=2*(size_t)nb*nb*sizeof(float);
    cudaFuncSetAttribute(diag_chol_inv2_k, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem);
    diag_chol_inv2_k<<<b,256,shmem>>>(D.data_ptr<float>(),L.data_ptr<float>(),Linv.data_ptr<float>(),nb);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"diag2 launch");
    return {L,Linv};
}
"""
K = load_inline(name="blocked_chol_v2", cpp_sources=["std::vector<torch::Tensor> diag_chol_inv2(torch::Tensor D);"],
                cuda_sources=[CUDA_SRC], functions=["diag_chol_inv2"], extra_cuda_cflags=["-O3"], verbose=False)


def blocked_chol(M, nb, tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    b, n, _ = M.shape
    L = torch.zeros_like(M)
    Mw = M.clone()
    for k in range(0, n, nb):
        kb = min(nb, n - k)
        D = Mw[:, k:k+kb, k:k+kb].contiguous()
        L11, L11inv = K.diag_chol_inv2(D)
        L[:, k:k+kb, k:k+kb] = L11
        if k + kb < n:
            M21 = Mw[:, k+kb:, k:k+kb].contiguous()
            L21 = torch.matmul(M21, L11inv.transpose(1, 2))
            L[:, k+kb:, k:k+kb] = L21
            Mw[:, k+kb:, k+kb:] -= torch.matmul(L21, L21.transpose(1, 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    return L


def _t(fn, r=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / r


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("== blocked Cholesky v2 (warp-col inverse): correctness + timing vs torch ==")
    for (B, N) in [(640, 512), (60, 1024), (8, 2048), (2, 4096)]:
        A = make_batch(B, N, 2, "dense", seed=0).cuda().contiguous()
        M = (torch.matmul(A.transpose(1, 2), A)).contiguous()
        M = M + (1e-2 * torch.diagonal(M, dim1=-2, dim2=-1).amax(-1)).view(-1, 1, 1) * torch.eye(N, device="cuda")
        Lref = torch.linalg.cholesky(M)
        t_torch = _t(lambda: torch.linalg.cholesky(M))
        best = None
        for nb in (64, 128, 256):
            if nb > N: continue
            try:
                Lb = blocked_chol(M, nb, tf32=True)
            except RuntimeError as e:
                print(f"  b{B} n{N} nb{nb}: launch fail ({str(e)[:30]})"); continue
            rec = (torch.matmul(Lb, Lb.transpose(1, 2)) - M).norm() / M.norm()
            t_blk = _t(lambda: blocked_chol(M, nb, tf32=True))
            D = M[:, :nb, :nb].contiguous(); t_diag = _t(lambda: K.diag_chol_inv2(D))
            spd = t_torch / t_blk
            if best is None or t_blk < best[1]: best = (nb, t_blk, spd)
            print(f"  b{B} n{N} nb{nb:3d}: recon {rec:.1e}  blocked {t_blk*1e3:7.1f}us  torch {t_torch*1e3:6.1f}us  {spd:.2f}x  diag1 {t_diag*1e3:5.1f}us")
        if best: print(f"     -> best nb{best[0]}: {best[1]*1e3:.1f}us ({best[2]:.2f}x torch)")
