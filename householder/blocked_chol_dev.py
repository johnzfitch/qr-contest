"""Milestone 1 of the CQR pivot: custom blocked tensor-core batched Cholesky.

torch.linalg.cholesky_ex = 3.9ms at n512 b640 (CUDA core). Target ~400us by doing the
O(n^3) bulk (panel TRSM + trailing SYRK) as batched tensor-core GEMMs, with only the
small nb x nb diagonal block factored+inverted in a custom per-matrix SMEM kernel (fp32).

Right-looking blocked Cholesky M = L L^T, block size nb:
  for each block-col k:
    L11,L11inv = diag_chol_inv(M[k,k])            # custom kernel, fp32, SMEM
    L21 = M[k+1:,k] @ L11inv^T                     # panel TRSM as GEMM (tensor core)
    M[k+1:,k+1:] -= L21 @ L21^T                    # trailing SYRK (tensor core, the bulk)

  source /workspace/qr/env.sh && python householder/blocked_chol_dev.py
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

// factor + invert an nb x nb SPD diagonal block, one CTA per matrix, fp32 in SMEM.
__global__ void diag_chol_inv_k(const float* __restrict__ Mblk, float* __restrict__ Lout,
                                float* __restrict__ Linvout, int nb){
    extern __shared__ float sm[];
    float* sL = sm; float* sX = sL + (size_t)nb*nb;
    int bid=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
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
    // inverse of lower-tri sL -> sX, one column per thread (columns independent)
    for(int col=tid; col<nb; col+=nt){
        for(int i=0;i<col;++i) sX[(size_t)i*nb+col]=0.f;
        for(int i=col;i<nb;++i){
            float s=(i==col)?1.f:0.f;
            for(int k=col;k<i;++k) s -= sL[(size_t)i*nb+k]*sX[(size_t)k*nb+col];
            sX[(size_t)i*nb+col] = s / sL[(size_t)i*nb+i];
        }
    }
    __syncthreads();
    for(int i=tid;i<nb*nb;i+=nt){ Lout[(size_t)bid*nb*nb+i]=sL[i]; Linvout[(size_t)bid*nb*nb+i]=sX[i]; }
}

std::vector<torch::Tensor> diag_chol_inv(torch::Tensor D){
    int b=D.size(0), nb=D.size(1);
    auto L=torch::empty_like(D), Linv=torch::empty_like(D);
    size_t shmem=2*(size_t)nb*nb*sizeof(float);
    cudaFuncSetAttribute(diag_chol_inv_k, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem);
    diag_chol_inv_k<<<b,256,shmem>>>(D.data_ptr<float>(),L.data_ptr<float>(),Linv.data_ptr<float>(),nb);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"diag launch");
    return {L,Linv};
}
"""
CPP_SRC = "std::vector<torch::Tensor> diag_chol_inv(torch::Tensor D);"
K = load_inline(name="blocked_chol_dev", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                functions=["diag_chol_inv"], extra_cuda_cflags=["-O3"], verbose=False)


def blocked_chol(M, nb, tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    b, n, _ = M.shape
    L = torch.zeros_like(M)
    Mw = M.clone()
    for k in range(0, n, nb):
        kb = min(nb, n - k)
        D = Mw[:, k:k+kb, k:k+kb].contiguous()
        L11, L11inv = K.diag_chol_inv(D)
        L[:, k:k+kb, k:k+kb] = L11
        if k + kb < n:
            M21 = Mw[:, k+kb:, k:k+kb].contiguous()
            L21 = torch.matmul(M21, L11inv.transpose(1, 2))   # M21 @ L11^-T
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
    print("== blocked Cholesky: correctness + timing + diag-kernel profile (vs torch) ==")
    for (B, N) in [(640, 512), (60, 1024), (8, 2048)]:
        A = make_batch(B, N, 2, "dense", seed=0).cuda().contiguous()
        M = (torch.matmul(A.transpose(1, 2), A)).contiguous()
        M = M + (1e-2 * torch.diagonal(M, dim1=-2, dim2=-1).amax(-1)).view(-1, 1, 1) * torch.eye(N, device="cuda")
        Lref = torch.linalg.cholesky(M)
        t_torch = _t(lambda: torch.linalg.cholesky(M))
        for nb in (32, 64, 128):
            Lb = blocked_chol(M, nb, tf32=True)
            err = (Lb - Lref).norm() / Lref.norm()
            rec = (torch.matmul(Lb, Lb.transpose(1, 2)) - M).norm() / M.norm()
            t_blk = _t(lambda: blocked_chol(M, nb, tf32=True))
            # diag-kernel-only cost (all block-cols' worth: one launch on the largest block, x nblocks approx)
            D = M[:, :nb, :nb].contiguous()
            t_diag1 = _t(lambda: K.diag_chol_inv(D))
            nblk = (N + nb - 1) // nb
            print(f"  b{B} n{N} nb{nb:3d}:  L-err {err:.1e} recon {rec:.1e}  "
                  f"blocked {t_blk*1e3:7.1f}us  torch {t_torch*1e3:6.1f}us  {t_torch/t_blk:.2f}x"
                  f"   diag1 {t_diag1*1e3:6.1f}us x{nblk}~{t_diag1*nblk*1e3:7.1f}us")
