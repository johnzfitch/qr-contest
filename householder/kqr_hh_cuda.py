"""Blocked WY-Householder QR — CUDA panel kernel + torch BLAS-3 trailing update.

THE NEW SPINE (replaces CholeskyQR2 + modified-LU reconstruction + theta-gate + geqrf
fallback). Produces the geqrf (H, tau) contract directly; unconditionally backward-stable,
so rankdef / clustered / nearrank pass with no special path. Math proven in kqr_hh.py.

Per panel [p0:pe] (width nb), one block per matrix:
  (1) panel_geqrt kernel (custom CUDA): unblocked geqr2 on the tall panel B[p0:n, p0:pe] in
      SMEM -> reflectors below diag (in B), R on/above diag (in B), tau[p0:pe], and the
      nb x nb WY block coefficient T (DLARFT). All in shared memory.
  (2) trailing update (torch / cuBLAS, BLAS-3, tensor-core):
          C = B[p0:n, pe:n] ;  V = unit-lower(B[p0:n, p0:pe])
          C -= V @ (Tᵀ @ (Vᵀ @ C))            # (I - V Tᵀ Vᵀ) C
Output H = B (reflectors below + R above), tau.

The kernel mirrors kqr_hh.hh_qr_blocked exactly (validated on CPU). This file cross-checks
the kernel against that torch mirror and the oracle checker, then benchmarks vs the current
CholeskyQR+modLU submission.

Run on the B200 pod:  python dev/kqr_hh_cuda.py
"""
import torch
from torch.utils.cpp_extension import load_inline
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from oracle import make_batch, check, SHAPES
from kqr_hh import hh_qr_blocked, dlarft  # CPU-validated mirror + T reference

torch.backends.cuda.matmul.allow_tf32 = False
EPS32 = torch.finfo(torch.float32).eps

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

/* block-wide sum reduction of `val`; result broadcast via sred[0]. nt = blockDim.x. */
__device__ float block_sum(float val, float* sred) {
    int tid = threadIdx.x, nt = blockDim.x;
    sred[tid] = val;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
    }
    float r = sred[0];
    __syncthreads();
    return r;
}

/* One block per matrix. Factor the tall panel B[p0:n, p0:pe] (m=n-p0 rows, nb cols) by
   unblocked Householder geqr2, then build the nb x nb WY T factor (DLARFT). */
__global__ void panel_geqrt_kernel(float* __restrict__ B, float* __restrict__ Tg,
                                   float* __restrict__ taug, int n, int p0, int nb) {
    extern __shared__ float smem[];
    int m = n - p0;
    float* sP   = smem;                 /* m*nb  panel (row-major: sP[r*nb+c]) */
    float* sT   = sP + (size_t)m * nb;  /* nb*nb T (row-major) */
    float* stau = sT + (size_t)nb * nb; /* nb    tau */
    float* sz   = stau + nb;            /* nb    DLARFT workspace */
    float* sred = sz + nb;              /* blockDim.x  reduction scratch */
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    float* Bb = B + (size_t)b * n * n;

    for (int idx = tid; idx < m * nb; idx += nt) {       /* load panel */
        int r = idx / nb, c = idx % nb;
        sP[idx] = Bb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < nb * nb; idx += nt) sT[idx] = 0.f;  /* T lower-tri stays 0 */
    __syncthreads();

    /* ---- unblocked geqr2 over the nb panel columns ---- */
    __shared__ float s_tau, s_denom;
    for (int j = 0; j < nb; ++j) {
        float part = 0.f;                                /* ||tail||^2 of column j */
        for (int r = j + 1 + tid; r < m; r += nt) { float v = sP[r * nb + j]; part += v * v; }
        float xn2 = block_sum(part, sred);
        if (tid == 0) {
            float alpha = sP[j * nb + j];
            float beta, tauj, denom;
            if (xn2 > 0.f) {
                float normf = sqrtf(alpha * alpha + xn2);
                beta  = (alpha >= 0.f) ? -normf : normf;
                tauj  = (beta - alpha) / beta;
                denom = alpha - beta;
            } else { beta = alpha; tauj = 0.f; denom = 1.f; }
            sP[j * nb + j] = beta;
            s_tau = tauj; s_denom = denom; stau[j] = tauj;
        }
        __syncthreads();
        float denom = s_denom, tauj = s_tau;
        for (int r = j + 1 + tid; r < m; r += nt) sP[r * nb + j] /= denom;   /* v = tail/denom */
        __syncthreads();
        /* apply H_j to trailing panel columns c in (j, nb): one thread per column */
        for (int c = j + 1 + tid; c < nb; c += nt) {
            float w = sP[j * nb + c];                    /* vfull[j]=1 */
            for (int r = j + 1; r < m; ++r) w += sP[r * nb + j] * sP[r * nb + c];
            w *= tauj;
            sP[j * nb + c] -= w;                          /* row j (vfull[j]=1) */
            for (int r = j + 1; r < m; ++r) sP[r * nb + c] -= sP[r * nb + j] * w;
        }
        __syncthreads();
    }

    /* ---- DLARFT: T (nb x nb upper-tri), forward columnwise ---- */
    for (int j = 0; j < nb; ++j) {
        if (j > 0) {
            /* z[i] = -tau_j * (V[:,i].V[:,j]) for i in [0,j) ; one thread per i */
            for (int i = tid; i < j; i += nt) {
                float d = sP[j * nb + i];                 /* r=j term: V[j,i]*1 */
                for (int r = j + 1; r < m; ++r) d += sP[r * nb + i] * sP[r * nb + j];
                sz[i] = -stau[j] * d;
            }
            __syncthreads();
            /* T[0:j, j] = T[0:j, 0:j] @ z[0:j]  (T upper-tri: T[i,k] for k>=i) */
            for (int i = tid; i < j; i += nt) {
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[i * nb + k] * sz[k];
                sT[i * nb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[j * nb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * nb; idx += nt) {        /* write panel back */
        int r = idx / nb, c = idx % nb;
        Bb[(size_t)(p0 + r) * n + (p0 + c)] = sP[idx];
    }
    for (int idx = tid; idx < nb * nb; idx += nt)         /* write T */
        Tg[(size_t)b * nb * nb + idx] = sT[idx];
    for (int i = tid; i < nb; i += nt)                    /* write tau */
        taug[(size_t)b * n + (p0 + i)] = stau[i];
}

void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb) {
    int batch = B.size(0), n = B.size(1);
    int m = n - p0;
    size_t shmem = ((size_t)m * nb + (size_t)nb * nb + 2 * nb + 256) * sizeof(float);
    cudaFuncSetAttribute(panel_geqrt_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_geqrt_kernel<<<batch, 256, shmem>>>(
        B.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), n, p0, nb);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
"""
CPP_SRC = "void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb);"

_K = load_inline(name="kqr_hh", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                 functions=["panel_geqrt"], extra_cuda_cflags=["-O3"], verbose=False)


def _nb_for(n):
    return max(8, min(64, 30000 // n))         # keep m*nb FP32 panel <= ~120KB SMEM


def hh_qr_cuda(A):
    b, n, _ = A.shape
    B = A.contiguous().clone()
    tau = torch.zeros(b, n, device=A.device)
    eye = torch.eye(n, device=A.device)
    p0 = 0
    while p0 < n:
        nb = min(_nb_for(n), n - p0)
        pe = p0 + nb
        T = torch.zeros(b, nb, nb, device=A.device)
        _K.panel_geqrt(B, T, tau, p0, nb)           # (1) panel factor -> reflectors, R, tau, T
        if pe < n:                                   # (2) WY trailing update (BLAS-3)
            panel = B[:, p0:, p0:pe]
            V = torch.tril(panel, -1)
            V[:, :nb, :nb] += eye[:nb, :nb]          # unit diagonal on the top block
            C = B[:, p0:, pe:]
            VtC = torch.einsum('bmi,bmc->bic', V, C)
            TtVtC = torch.einsum('bki,bkc->bic', T, VtC)
            B[:, p0:, pe:] = C - torch.einsum('bmi,bic->bmc', V, TtVtC)
        p0 = pe
    return B, tau


# --------------------------------------------------------------------------- #
# Validation + benchmark.                                                      #
# --------------------------------------------------------------------------- #
def _report(tag, A, H, tau, ref=None):
    fr, og, ft, ot, ps = check(A, H, tau)
    extra = ""
    if ref is not None:
        Hr, tr = ref
        extra = f"  dH={(H - Hr).abs().max().item():.2e} dtau={(tau - tr).abs().max().item():.2e}"
    flag = "OK  " if ps.all() else "FAIL"
    print(f"  {tag:24s} factor={fr:.2e}/{ft:.2e} ortho={og:.2e}/{ot:.2e} {flag} "
          f"({int(ps.sum())}/{len(ps)}){extra}")
    return bool(ps.all())


def _bench(fn, iters=5):
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms


def main():
    assert torch.cuda.is_available(), "run on the B200 pod"
    print("blocked WY-Householder CUDA kernel — validation vs CPU mirror + oracle\n")

    # correctness on small batches first (cross-check kernel vs torch mirror)
    print("--- correctness (kernel vs hh_qr_blocked mirror) ---")
    allok = True
    for (b, n, cond, case) in SHAPES:
        bb = min(b, 8)
        A = make_batch(bb, n, cond, case, seed=0)
        H, tau = hh_qr_cuda(A)
        Hr, tr = hh_qr_blocked(A, nb=_nb_for(n))
        allok &= _report(f"b{bb}_n{n}_{case}", A, H, tau, ref=(Hr, tr))
    print(f"\nALL PASS: {allok}\n")

    # full-batch benchmark vs geqrf baseline
    print("--- benchmark (full batch) vs torch.geqrf ---")
    geo_ours, geo_ref = 1.0, 1.0
    for (b, n, cond, case) in SHAPES:
        A = make_batch(b, n, cond, case, seed=0)
        t_ours = _bench(lambda: hh_qr_cuda(A))
        t_ref = _bench(lambda: torch.geqrf(A))
        geo_ours *= t_ours; geo_ref *= t_ref
        print(f"  b{b}_n{n}_{case:9s}: ours={t_ours:8.2f}ms  geqrf={t_ref:8.2f}ms  "
              f"{t_ref / t_ours:5.2f}x")
    k = len(SHAPES)
    print(f"\ngeomean ours={geo_ours ** (1/k):.2f}ms  geqrf={geo_ref ** (1/k):.2f}ms")


if __name__ == "__main__":
    main()
