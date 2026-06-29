"""btrail_derisk — B GATE: can a hand-written in-kernel WY trailing GEMM stay
within ~2x of cuBLAS at::matmul on the n512 b640 step shapes?

This is the load-bearing risk for Option B (persistent lookahead kernel): the
trailing CTAs cannot call cuBLAS, so they must hand-roll the WY update
    W = Vt C ;  W = Tt W ;  C -= V W
If this is >~2x slower than at::matmul, the lookahead overlap can't pay it back
and B is dead. Build the op, validate it, time it vs cuBLAS over the FULL n512
trailing step sequence (the same shapes qr_blocked feeds).

  source /workspace/qr/env.sh && python householder/btrail_derisk.py
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
#include <algorithm>

#define TN 64    /* C columns per CTA */

/* One CTA per (matrix b, col-tile). V staged in SMEM (reused both steps + all TN
   cols); T staged in SMEM; C streamed from global. Computes, in place:
       W  = Vt C          (nb x tn)
       W2 = Tt W          (nb x tn, T upper-tri DLARFT)
       C -= V W2          (m x tn)
   V is the EXPLICIT reflector tile (diag=1, below=tail, above=0). */
__global__ void btrail_kernel(const float* __restrict__ Vg, const float* __restrict__ Tg,
                              float* __restrict__ Cg, int m, int nb, int ncol) {
    extern __shared__ float smem[];
    float* sV  = smem;                 /* m  * nb */
    float* sT  = sV  + (size_t)m * nb; /* nb * nb */
    float* sW  = sT  + (size_t)nb * nb;/* nb * TN */
    float* sW2 = sW  + (size_t)nb * TN;/* nb * TN */
    int b = blockIdx.x, c0 = blockIdx.y * TN;
    int tn = min(TN, ncol - c0);
    int tid = threadIdx.x, nt = blockDim.x;
    const float* Vb = Vg + (size_t)b * m * nb;
    const float* Tb = Tg + (size_t)b * nb * nb;
    float* Cb = Cg + (size_t)b * m * ncol;

    for (int idx = tid; idx < m * nb;  idx += nt) sV[idx] = Vb[idx];
    for (int idx = tid; idx < nb * nb; idx += nt) sT[idx] = Tb[idx];
    __syncthreads();

    /* Step1: W[i][c] = sum_r V[r][i] * C[r][c0+c] */
    for (int idx = tid; idx < nb * tn; idx += nt) {
        int i = idx / tn, c = idx % tn;
        float acc = 0.f;
        for (int r = 0; r < m; ++r) acc += sV[(size_t)r * nb + i] * Cb[(size_t)r * ncol + (c0 + c)];
        sW[(size_t)i * TN + c] = acc;
    }
    __syncthreads();

    /* Step2: W2[i][c] = sum_{k<=i} T[k][i] * W[k][c]   (Tt, upper-tri) */
    for (int idx = tid; idx < nb * tn; idx += nt) {
        int i = idx / tn, c = idx % tn;
        float acc = 0.f;
        for (int k = 0; k <= i; ++k) acc += sT[(size_t)k * nb + i] * sW[(size_t)k * TN + c];
        sW2[(size_t)i * TN + c] = acc;
    }
    __syncthreads();

    /* Step3: C[r][c0+c] -= sum_i V[r][i] * W2[i][c] */
    for (int idx = tid; idx < m * tn; idx += nt) {
        int r = idx / tn, c = idx % tn;
        float acc = 0.f;
        for (int i = 0; i < nb; ++i) acc += sV[(size_t)r * nb + i] * sW2[(size_t)i * TN + c];
        Cb[(size_t)r * ncol + (c0 + c)] -= acc;
    }
}

void btrail(torch::Tensor V, torch::Tensor T, torch::Tensor C) {
    int b = V.size(0), m = V.size(1), nb = V.size(2), ncol = C.size(2);
    dim3 grid(b, (ncol + TN - 1) / TN);
    size_t shmem = ((size_t)m * nb + (size_t)nb * nb + 2 * (size_t)nb * TN) * sizeof(float);
    cudaFuncSetAttribute(btrail_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    btrail_kernel<<<grid, 256, shmem>>>(V.data_ptr<float>(), T.data_ptr<float>(),
                                        C.data_ptr<float>(), m, nb, ncol);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "btrail launch failed");
}
"""
CPP_SRC = "void btrail(torch::Tensor V, torch::Tensor T, torch::Tensor C);"

K = load_inline(name="btrail_derisk", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                functions=["btrail"], extra_cuda_cflags=["-O3"], verbose=False)


def ref_trailing(V, T, C):
    """cuBLAS WY trailing, out-of-place: returns updated C."""
    W = torch.matmul(V.transpose(1, 2), C)
    W = torch.matmul(T.transpose(1, 2), W)
    return C - torch.matmul(V, W)


def make_step(b, m, nb, ncol, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    V = torch.randn(b, m, nb, device="cuda", generator=g)
    # explicit reflector tile: diag=1, above=0, below=tail (kqr_fused_v2 layout)
    V = torch.tril(V, -1)
    eye = torch.zeros(b, m, nb, device="cuda")
    for i in range(min(m, nb)):
        eye[:, i, i] = 1.0
    V = V + eye
    T = torch.triu(torch.randn(b, nb, nb, device="cuda", generator=g))
    C = torch.randn(b, m, ncol, device="cuda", generator=g)
    return V.contiguous(), T.contiguous(), C.contiguous()


def validate():
    print("== btrail correctness ==")
    worst = 0.0
    for (m, nb, ncol) in [(512, 32, 480), (256, 32, 224), (512, 32, 64), (128, 32, 96), (512, 16, 480)]:
        V, T, C = make_step(8, m, nb, ncol, seed=1)
        Cref = ref_trailing(V, T, C)
        Cmine = C.clone()
        K.btrail(V, T, Cmine)
        rel = (Cmine - Cref).abs().max() / (Cref.abs().max() + 1e-12)
        worst = max(worst, rel.item())
        print(f"  m={m:<4d} nb={nb:<3d} ncol={ncol:<4d}  rel={rel.item():.2e}")
    print(f"  worst rel = {worst:.2e}  ({'OK' if worst < 1e-4 else 'FAIL'})")
    return worst


def _evt():
    return torch.cuda.Event(enable_timing=True)


def _time(fn, reps=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def timing(n=512, b=640, nb=32):
    print(f"\n== btrail timing: full n={n} b={b} trailing step sequence (nb={nb}) ==")
    steps = []
    p0 = 0
    while p0 < n:
        kb = min(nb, n - p0); pe = p0 + kb
        if pe < n:
            steps.append((n - p0, kb, n - pe))   # (m, nb, ncol)
        p0 = pe
    tensors = [make_step(b, m, k, ncol, seed=p) for p, (m, k, ncol) in enumerate(steps)]

    def run_cublas():
        for (V, T, C) in tensors:
            W = torch.matmul(V.transpose(1, 2), C); W = torch.matmul(T.transpose(1, 2), W); C.sub_(torch.matmul(V, W))

    def run_kernel():
        for (V, T, C) in tensors:
            K.btrail(V, T, C)

    tc = _time(run_cublas)
    tk = _time(run_kernel)
    print(f"  steps={len(steps)}  cuBLAS={tc:.3f}ms  hand-kernel={tk:.3f}ms  ratio={tk/tc:.2f}x")
    print(f"  >>> {'VIABLE for B (<2x)' if tk/tc < 2.0 else 'AT RISK (>2x) — B trailing needs work'}")
    return tk / tc


if __name__ == "__main__":
    assert torch.cuda.is_available()
    if validate() < 1e-4:
        timing(512, 640); timing(1024, 60); timing(352, 40)
