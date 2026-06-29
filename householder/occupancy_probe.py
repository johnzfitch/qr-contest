"""Occupancy probe: how many CTAs/SM does the fused panel get at n=512, and is it
SMEM-bound? cudaOccupancyMaxActiveBlocksPerMultiprocessor (host API, no ncu needed).
Correlate with n512 timing across nb: if occupancy-bound, smaller nb (less SMEM ->
more CTAs/SM) should run FASTER despite more panel steps.

  source /workspace/qr/env.sh && python householder/occupancy_probe.py
"""
import sys, pathlib, importlib.util
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch        # noqa: E402

# minimal: compile the v3 panel kernel + an occupancy query against it.
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ float warp_sum(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffu,v,o);return v;}
__device__ __forceinline__ float warp_max(float v){for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));return v;}

/* same body as panel_geqrt_v3_kernel (regs/smem identical -> valid occupancy query) */
__global__ void panel_k(float* __restrict__ Hm, float* __restrict__ Tg, float* __restrict__ taug,
                        float* __restrict__ Vout, int n, int p0, int kb, int ldt, int ldv){
    extern __shared__ float smem[];
    int m=n-p0, LD=kb+1; float* sP=smem; float* sT=sP+(size_t)m*LD; float* stau=sT+(size_t)kb*kb; float* sz=stau+kb;
    int b=blockIdx.x,tid=threadIdx.x,nt=blockDim.x; int warp=tid>>5,lane=tid&31,nwarp=nt>>5;
    float* Hb=Hm+(size_t)b*n*n; __shared__ float s_tau,s_denom;
    for(int idx=tid;idx<m*kb;idx+=nt){int r=idx/kb,c=idx%kb; sP[(size_t)r*LD+c]=Hb[(size_t)(p0+r)*n+(p0+c)];}
    for(int idx=tid;idx<kb*kb;idx+=nt)sT[idx]=0.f; __syncthreads();
    for(int j=0;j<kb;++j){
        if(warp==0){float amax=0.f;for(int r=j+1+lane;r<m;r+=32)amax=fmaxf(amax,fabsf(sP[(size_t)r*LD+j]));amax=warp_max(amax);
            float ssq=0.f; if(amax>0.f)for(int r=j+1+lane;r<m;r+=32){float t=sP[(size_t)r*LD+j]/amax;ssq+=t*t;} ssq=warp_sum(ssq);
            float xnorm=(amax>0.f)?amax*sqrtf(ssq):0.f;
            if(lane==0){float alpha=sP[(size_t)j*LD+j],beta,tauj,denom; if(xnorm==0.f){beta=alpha;tauj=0.f;denom=1.f;}else{beta=-copysignf(hypotf(alpha,xnorm),alpha);tauj=(beta-alpha)/beta;denom=alpha-beta;} sP[(size_t)j*LD+j]=beta;stau[j]=tauj;s_tau=tauj;s_denom=denom;}
            __syncwarp(); float denom=s_denom; for(int r=j+1+lane;r<m;r+=32)sP[(size_t)r*LD+j]/=denom;}
        __syncthreads(); float tauj=s_tau;
        for(int c=j+1+warp;c<kb;c+=nwarp){float partial=0.f;for(int r=j+1+lane;r<m;r+=32)partial+=sP[(size_t)r*LD+j]*sP[(size_t)r*LD+c];
            float w=(warp_sum(partial)+sP[(size_t)j*LD+c])*tauj; if(lane==0)sP[(size_t)j*LD+c]-=w; for(int r=j+1+lane;r<m;r+=32)sP[(size_t)r*LD+c]-=sP[(size_t)r*LD+j]*w;}
        __syncthreads();}
    for(int j=0;j<kb;++j){if(j>0){for(int i=warp;i<j;i+=nwarp){float d=0.f;for(int r=j+1+lane;r<m;r+=32)d+=sP[(size_t)r*LD+i]*sP[(size_t)r*LD+j];d=warp_sum(d);if(lane==0)sz[i]=-stau[j]*(d+sP[(size_t)j*LD+i]);}
        __syncthreads(); for(int i=tid;i<j;i+=nt){float acc=0.f;for(int k=i;k<j;++k)acc+=sT[(size_t)i*kb+k]*sz[k];sT[(size_t)i*kb+j]=acc;} __syncthreads();}
        if(tid==0)sT[(size_t)j*kb+j]=stau[j]; __syncthreads();}
    for(int idx=tid;idx<m*kb;idx+=nt){int r=idx/kb,c=idx%kb;float val=sP[(size_t)r*LD+c];Hb[(size_t)(p0+r)*n+(p0+c)]=val;float vv=(r==c)?1.f:((r>c)?val:0.f);Vout[(size_t)b*n*ldv+(size_t)(p0+r)*ldv+c]=vv;}
    for(int i=tid;i<kb;i+=nt)for(int j=0;j<kb;++j)Tg[(size_t)b*ldt*ldt+(size_t)i*ldt+j]=sT[(size_t)i*kb+j];
    for(int i=tid;i<kb;i+=nt)taug[(size_t)b*n+(p0+i)]=stau[i];
}

std::vector<int64_t> occupancy(int64_t n, int64_t nb){
    int m=n, LD=nb+1;
    size_t shmem=((size_t)m*LD+(size_t)nb*nb+2*nb+32)*sizeof(float);
    cudaFuncSetAttribute(panel_k, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem);
    int maxb=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, panel_k, 256, shmem);
    cudaFuncAttributes a; cudaFuncGetAttributes(&a, panel_k);
    int dev; cudaGetDevice(&dev); int smem_sm=0,nsm=0;
    cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    return {maxb,(int64_t)shmem,(int64_t)a.numRegs,(int64_t)smem_sm,(int64_t)nsm};
}
"""
CPP_SRC = "std::vector<int64_t> occupancy(int64_t n, int64_t nb);"
K = load_inline(name="occ_probe", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                functions=["occupancy"], extra_cuda_cflags=["-O3"], verbose=False)

v3 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v3", QRPY/"householder"/"kqr_fused_v3.py"))
importlib.util.spec_from_file_location("v3", QRPY/"householder"/"kqr_fused_v3.py").loader.exec_module(v3)


def _t(fn, A, r=30):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn(A)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/r


if __name__ == "__main__":
    assert torch.cuda.is_available()
    A = make_batch(640, 512, 2, "dense", seed=0).cuda().contiguous()
    print(f"{'n':>5s}{'nb':>4s} {'maxBlk/SM':>10s} {'smem/CTA(KB)':>13s} {'regs':>5s} {'smemSM(KB)':>11s} {'n512 ms':>9s}")
    for nb in (8, 16, 24, 32):
        occ = K.occupancy(512, nb)
        maxb, shmem, regs, smemsm, nsm = occ
        t = _t(lambda x: v3.custom_kernel(x, nb), A)
        print(f"{512:>5d}{nb:>4d} {maxb:>10d} {shmem/1024:>13.1f} {regs:>5d} {smemsm/1024:>11.1f} {t:>9.3f}")
    print(f"(SMs={nsm})  -- if maxBlk/SM rises as nb shrinks AND n512 ms drops, panel is OCCUPANCY-bound.")
