"""large_n_cluster_probe — THE GATE for the cluster-per-matrix panel.

The n2048 panel is 39 ms (84%) at 5.4% device fill. Multi-CTA-per-matrix would
split the panel's m-row reductions across N cooperating CTAs. The make-or-break
unknown (same role btrail played for B): the CROSS-CTA REDUCTION COST. We isolate it.

Headline metric: RN/R1 = (N-CTA cooperative reduction over m) / (single-CTA reduction
over m), at the panel's m. The panel is reduction-bound, so:
   RN/R1 < 1  -> row-split WINS; projected panel ~= 39 ms * RN/R1  -> build the cluster panel.
   RN/R1 >= 1 -> cross-CTA overhead eats the parallelism -> cluster path DEAD, ship re-route.

Two cross-CTA mechanisms (chainsmoker's "cluster vs plain-grid"):
   (a) cluster DSMEM (__cluster_dims__ + map_shared_rank + cluster.sync) — the fast path.
   (b) plain-grid cooperative (cudaLaunchCooperativeKernel + grid.sync + global atomic).
Plus: TF32-trailing re-check at the cond-1 large-n shapes (probe#4 killed TF32 on cond-2/band;
the board large-n shapes are cond-1 dense, so re-test).

  source /workspace/qr/env.sh && python householder/large_n_cluster_probe.py
"""
import sys, pathlib, importlib.util
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve()
QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY / "common")]
from oracle import make_batch, check        # noqa: E402

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define FULL 0xffffffffu

__device__ __forceinline__ float block_reduce(float v, float* sm) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    float t = (tid < nw) ? sm[tid] : 0.f;
    if (wid == 0) { for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(FULL, t, o); if (lane == 0) sm[0] = t; }
    __syncthreads();
    return sm[0];   /* sm[0] holds block total, readable cross-CTA via DSMEM */
}

/* N-CTA cooperative ssq-reduction over x[0:m], split by block_rank, combined via
   cluster DSMEM. Loops `reps` reductions to isolate steady-state per-reduction cost. */
template<int CS>
__global__ void __cluster_dims__(CS, 1, 1)
cluster_reduce_kernel(const float* __restrict__ x, float* __restrict__ out, int m, int reps) {
    cg::cluster_group cl = cg::this_cluster();
    int rank = cl.block_rank(), tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ float sm[];
    int chunk = (m + CS - 1) / CS, lo = rank * chunk, hi = min(m, lo + chunk);
    float acc = 0.f;
    for (int rep = 0; rep < reps; ++rep) {
        float partial = 0.f;
        for (int i = lo + tid; i < hi; i += nt) { float v = x[i] + rep * 1e-30f; partial += v * v; }
        block_reduce(partial, sm);          /* sm[0] = this CTA's partial */
        cl.sync();
        float total = 0.f;
        for (int r = 0; r < CS; ++r) total += *cl.map_shared_rank(sm, r);   /* DSMEM gather */
        cl.sync();
        acc += total;
    }
    if (tid == 0 && rank == 0) out[blockIdx.x / CS] = acc;
}

/* plain-grid cooperative reduction: N blocks, global-atomic combine + grid.sync. */
__global__ void grid_reduce_kernel(const float* __restrict__ x, float* __restrict__ gacc, int m, int reps) {
    cg::grid_group grid = cg::this_grid();
    int N = gridDim.x, rank = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ float sm[];
    int chunk = (m + N - 1) / N, lo = rank * chunk, hi = min(m, lo + chunk);
    float acc = 0.f;
    for (int rep = 0; rep < reps; ++rep) {
        if (rank == 0 && tid == 0) *gacc = 0.f;
        grid.sync();
        float partial = 0.f;
        for (int i = lo + tid; i < hi; i += nt) { float v = x[i] + rep * 1e-30f; partial += v * v; }
        partial = block_reduce(partial, sm);
        if (tid == 0) atomicAdd(gacc, partial);
        grid.sync();
        acc += *gacc;
        grid.sync();
    }
    if (rank == 0 && tid == 0) gacc[1] = acc;
}

template<int CS>
static float run_cluster(torch::Tensor x, int reps, int num_groups) {
    int m = x.size(0);
    auto out = torch::zeros({num_groups}, x.options());
    size_t shmem = 64 * sizeof(float);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cluster_reduce_kernel<CS><<<num_groups * CS, 256, shmem>>>(x.data_ptr<float>(), out.data_ptr<float>(), m, 3);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    cluster_reduce_kernel<CS><<<num_groups * CS, 256, shmem>>>(x.data_ptr<float>(), out.data_ptr<float>(), m, reps);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cluster launch failed");
    return ms / reps;   /* ms per reduction */
}

static float run_grid(torch::Tensor x, int reps, int N) {
    int m = x.size(0);
    auto gacc = torch::zeros({2}, x.options());
    size_t shmem = 64 * sizeof(float);
    void* k = (void*)grid_reduce_kernel;
    float* xp = x.data_ptr<float>(); float* gp = gacc.data_ptr<float>();
    int warm = 3;
    void* args[] = {&xp, &gp, &m, &warm};
    cudaLaunchCooperativeKernel(k, dim3(N), dim3(256), args, shmem);
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    void* args2[] = {&xp, &gp, &m, &reps};
    cudaEventRecord(a);
    cudaLaunchCooperativeKernel(k, dim3(N), dim3(256), args2, shmem);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "grid coop launch failed");
    return ms / reps;
}

/* dispatchers */
float cluster_reduce(torch::Tensor x, int64_t CS, int64_t reps, int64_t num_groups) {
    if (CS == 1) return run_cluster<1>(x, reps, num_groups);
    if (CS == 2) return run_cluster<2>(x, reps, num_groups);
    if (CS == 4) return run_cluster<4>(x, reps, num_groups);
    if (CS == 8) return run_cluster<8>(x, reps, num_groups);
    TORCH_CHECK(false, "CS must be 1/2/4/8"); return 0.f;
}
float grid_reduce(torch::Tensor x, int64_t N, int64_t reps) { return run_grid(x, reps, (int)N); }
"""
CPP_SRC = ("float cluster_reduce(torch::Tensor x, int64_t CS, int64_t reps, int64_t num_groups);\n"
           "float grid_reduce(torch::Tensor x, int64_t N, int64_t reps);")

K = load_inline(name="large_n_cluster_probe", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                functions=["cluster_reduce", "grid_reduce"], extra_cuda_cflags=["-O3"], verbose=False)

fused = None


def _imp(p):
    s = importlib.util.spec_from_file_location(p.stem, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def reduction_battery():
    print("== CROSS-CTA REDUCTION COST (THE GATE) ==")
    print("   per-reduction us; RN/R1 = N-CTA / single-CTA; project panel = 39ms * RN/R1\n")
    for m in (2048, 4096):
        x = torch.randn(m, device="cuda")
        ng = 8 if m == 2048 else 2          # mimic the real batch (b8 / b2)
        r1 = K.cluster_reduce(x, 1, 4000, ng) * 1e3
        print(f"  m={m} (groups={ng}):  single-CTA R1 = {r1:.3f} us")
        for CS in (2, 4, 8):
            rn = K.cluster_reduce(x, CS, 4000, ng) * 1e3
            proj = 39.0 * (rn / r1) if m == 2048 else None
            tag = f"  -> proj n2048 panel ~= {proj:.1f} ms" if proj else ""
            print(f"      cluster-{CS} DSMEM   RN = {rn:6.3f} us   RN/R1 = {rn/r1:.2f}{tag}")
        for N in (2, 4, 8):
            try:
                rg = K.grid_reduce(x, N, 4000) * 1e3
                print(f"      grid-{N} coop       RG = {rg:6.3f} us   RG/R1 = {rg/r1:.2f}")
            except Exception as ex:
                print(f"      grid-{N} coop       FAILED: {ex}")
        print()


def tf32_recheck():
    print("== TF32-trailing re-check at cond-1 large-n (probe#4 killed it on cond-2/band) ==")
    global fused
    fused = _imp(QRPY / "householder" / "kqr_fused_v2.py")
    for (b, n) in [(8, 2048), (2, 4096)]:
        A = make_batch(b, n, 1, "dense", seed=0).cuda().contiguous()
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            H, tau = fused.custom_kernel(A, 32)
            fr, og, ft, ot, ps = check(A, H, tau)
            margin = max(fr / ft, og / ot)
            # time
            for _ in range(3): fused.custom_kernel(A, 32)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10): fused.custom_kernel(A, 32)
            e.record(); torch.cuda.synchronize(); t = s.elapsed_time(e) / 10
            print(f"  n={n} b={b} tf32={int(tf32)}: pass={int(ps.sum())}/{b} margin={margin:.4f}  {t:.2f} ms")
        torch.backends.cuda.matmul.allow_tf32 = False
    print()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    reduction_battery()
    tf32_recheck()
    print("READ: if any cluster-N RN/R1 < 1 at m=2048 -> build cluster panel (projected panel above).")
    print("      if all RN/R1 >= 1 -> cross-CTA overhead dominates -> ship #7+routing+reroute.")
