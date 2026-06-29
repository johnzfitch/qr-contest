"""Phase-2 CLUSTER / DSMEM bring-up for the fused KAN kernel (n >= 256).

The single-block fused kernel (dev/kqr_fused.py) tops out at n<=~168 because one
block's 227 KB SMEM can't hold the working set for larger n. For n in {352, 512}
the whole 2*n^2 working set DOES fit across a THREAD-BLOCK CLUSTER's distributed
shared memory (DSMEM): any block in the cluster can read/write any other block's
SMEM, so C blocks give C*227 KB of addressable on-chip storage.

This file is the FIRST step: a minimal probe that de-risks the cluster API on the
B200 (sm_100, CUDA 13) before we port the KAN pipeline onto it:
  * launch with a cluster dimension via cudaLaunchKernelEx,
  * cg::this_cluster(): block_rank(), num_blocks(), sync(),
  * cluster.map_shared_rank(ptr, rank): read a *neighbor block's* SMEM (DSMEM),
  * non-portable cluster size > 8 (opt-in) so n=512 (needs ~11-16 blocks) works.

Run on the Runpod B200:  python dev/kqr_cluster.py
"""
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
namespace cg = cooperative_groups;

/* Each block stores its GLOBAL block id in its own SMEM[0]. After a cluster
   barrier, every block reads the SMEM[0] of its in-cluster neighbor (rank+1)
   via distributed shared memory, and writes that value to out[global_block_id].
   Correct DSMEM => out[bid] == (global id of neighbor) for every block. */
__global__ void cluster_probe_kernel(int* __restrict__ out) {
    extern __shared__ int sm[];
    cg::cluster_group cluster = cg::this_cluster();
    unsigned rank = cluster.block_rank();
    unsigned nb   = cluster.num_blocks();
    if (threadIdx.x == 0) sm[0] = (int)blockIdx.x;       /* publish my global id */
    cluster.sync();                                       /* all blocks have written */
    if (threadIdx.x == 0) {
        unsigned nbr = (rank + 1u) % nb;
        int* remote  = cluster.map_shared_rank(sm, nbr); /* neighbor's SMEM window */
        out[blockIdx.x] = remote[0];
    }
    cluster.sync();                                       /* don't exit before reads done */
}

torch::Tensor cluster_probe(int num_clusters, int C) {
    int total = num_clusters * C;
    auto out = torch::full({total}, -1,
                           torch::dtype(torch::kInt32).device(torch::kCUDA));
    size_t smem = 128;                                    /* bytes; tiny */
    cudaFuncSetAttribute(cluster_probe_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (C > 8) {
        cudaError_t ep = cudaFuncSetAttribute(
            cluster_probe_kernel,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        TORCH_CHECK(ep == cudaSuccess,
                    "non-portable cluster opt-in: ", cudaGetErrorString(ep));
    }
    cudaLaunchConfig_t config = {};
    config.gridDim         = dim3(total, 1, 1);
    config.blockDim        = dim3(32, 1, 1);
    config.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr[1];
    attr[0].id                 = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x   = C;
    attr[0].val.clusterDim.y   = 1;
    attr[0].val.clusterDim.z   = 1;
    config.attrs    = attr;
    config.numAttrs = 1;
    cudaError_t e = cudaLaunchKernelEx(&config, cluster_probe_kernel,
                                       out.data_ptr<int>());
    TORCH_CHECK(e == cudaSuccess, "launch C=", C, ": ", cudaGetErrorString(e));
    e = cudaDeviceSynchronize();
    TORCH_CHECK(e == cudaSuccess, "sync C=", C, ": ", cudaGetErrorString(e));
    return out;
}
"""
CPP_SRC = "torch::Tensor cluster_probe(int num_clusters, int C);"


def main():
    mod = load_inline(
        name="kqr_cluster_probe", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
        functions=["cluster_probe"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_100,code=sm_100",
                           "--expt-relaxed-constexpr"], verbose=True)
    print("cluster DSMEM probe (out[bid] should == neighbor's global block id)\n")
    NC = 3
    for C in (2, 4, 8, 11, 16):
        out = mod.cluster_probe(NC, C).cpu()
        bids = torch.arange(NC * C)
        rank = bids % C
        base = bids - rank
        expected = (base + (rank + 1) % C).to(torch.int32)
        ok = torch.equal(out, expected)
        print(f"  C={C:2d}  cluster.sync+map_shared_rank: {'OK' if ok else 'FAIL'}"
              + ("" if ok else f"  got={out.tolist()}\n      exp={expected.tolist()}"))


if __name__ == "__main__":
    main()
