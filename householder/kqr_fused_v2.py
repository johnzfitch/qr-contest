"""kqr_fused_v2 — PHASE 2: fuse the blocked outer loop behind ONE C++ entry.

Phase-1 showed trailing+orch costs ~0.55 ms/panel independent of nb, while the actual
GEMM compute is ~30 us/panel -> it is per-panel OVERHEAD (python launch latency +
tril/eye/zeros allocs), not arithmetic. This removes all of it:
  * qr_blocked(A, NB): the whole panel loop runs in C++ (no python per-panel).
  * the panel kernel writes the EXPLICIT V tile (diag=1, above=0, below=tail) into a
    persistent workspace -> no torch.tril / no identity add in the timed path.
  * T is written into a persistent buffer -> no torch.zeros(T) per panel.
  * trailing update via at::matmul on the current device queue, same fp32 WY:
        W = Vᵀ C ; W = Tᵀ W ; C -= V W.
Same geqrf-compatible (H, tau) output. Correctness identical to kqr_panel_v2.

  source /workspace/qr/env.sh && python householder/kqr_fused_v2.py
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
#include <vector>
#include <algorithm>

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* One CTA per matrix. Factor panel H[p0:n, p0:p0+kb], write compact (H,tau)+T, and
   the EXPLICIT V tile (diag=1, above=0, below=tail) into Vout[:, p0:n, 0:kb]. */
__global__ void panel_geqrt_v2_kernel(float* __restrict__ Hm, float* __restrict__ Tg,
                                      float* __restrict__ taug, float* __restrict__ Vout,
                                      int n, int p0, int kb, int ldt, int ldv) {
    extern __shared__ float smem[];
    int m = n - p0, LD = kb + 1;
    float* sP   = smem;
    float* sT   = sP + (size_t)m * LD;
    float* stau = sT + (size_t)kb * kb;
    float* sz   = stau + kb;
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
    float* Hb = Hm + (size_t)b * n * n;
    __shared__ float s_tau, s_denom;

    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, c = idx % kb;
        sP[(size_t)r * LD + c] = Hb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < kb * kb; idx += nt) sT[idx] = 0.f;
    __syncthreads();

    for (int j = 0; j < kb; ++j) {
        if (warp == 0) {
            float amax = 0.f;
            for (int r = j + 1 + lane; r < m; r += 32) amax = fmaxf(amax, fabsf(sP[(size_t)r * LD + j]));
            amax = warp_max(amax);
            float ssq = 0.f;
            if (amax > 0.f) for (int r = j + 1 + lane; r < m; r += 32) { float t = sP[(size_t)r * LD + j] / amax; ssq += t * t; }
            ssq = warp_sum(ssq);
            float xnorm = (amax > 0.f) ? amax * sqrtf(ssq) : 0.f;
            if (lane == 0) {
                float alpha = sP[(size_t)j * LD + j], beta, tauj, denom;
                if (xnorm == 0.f) { beta = alpha; tauj = 0.f; denom = 1.f; }
                else { beta = -copysignf(hypotf(alpha, xnorm), alpha); tauj = (beta - alpha) / beta; denom = alpha - beta; }
                sP[(size_t)j * LD + j] = beta; stau[j] = tauj; s_tau = tauj; s_denom = denom;
            }
            __syncwarp();
            float denom = s_denom;
            for (int r = j + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + j] /= denom;
        }
        __syncthreads();
        float tauj = s_tau;
        for (int c = j + 1 + warp; c < kb; c += nwarp) {
            float partial = 0.f;
            for (int r = j + 1 + lane; r < m; r += 32) partial += sP[(size_t)r * LD + j] * sP[(size_t)r * LD + c];
            float w = (warp_sum(partial) + sP[(size_t)j * LD + c]) * tauj;
            if (lane == 0) sP[(size_t)j * LD + c] -= w;
            for (int r = j + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + c] -= sP[(size_t)r * LD + j] * w;
        }
        __syncthreads();
    }

    for (int j = 0; j < kb; ++j) {                  /* DLARFT T */
        if (j > 0) {
            for (int i = tid; i < j; i += nt) {
                float d = sP[(size_t)j * LD + i];
                for (int r = j + 1; r < m; ++r) d += sP[(size_t)r * LD + i] * sP[(size_t)r * LD + j];
                sz[i] = -stau[j] * d;
            }
            __syncthreads();
            for (int i = tid; i < j; i += nt) {
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[(size_t)i * kb + k] * sz[k];
                sT[(size_t)i * kb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[(size_t)j * kb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * kb; idx += nt) {   /* compact H + explicit V */
        int r = idx / kb, c = idx % kb;
        float val = sP[(size_t)r * LD + c];
        Hb[(size_t)(p0 + r) * n + (p0 + c)] = val;
        float vv = (r == c) ? 1.f : ((r > c) ? val : 0.f);
        Vout[(size_t)b * n * ldv + (size_t)(p0 + r) * ldv + c] = vv;
    }
    for (int i = tid; i < kb; i += nt)
        for (int j = 0; j < kb; ++j) Tg[(size_t)b * ldt * ldt + (size_t)i * ldt + j] = sT[(size_t)i * kb + j];
    for (int i = tid; i < kb; i += nt) taug[(size_t)b * n + (p0 + i)] = stau[i];
}

static void launch_panel(torch::Tensor H, torch::Tensor T, torch::Tensor tau,
                         torch::Tensor V, int p0, int kb) {
    int batch = H.size(0), n = H.size(1), m = n - p0, LD = kb + 1;
    int ldt = T.size(2), ldv = V.size(2);
    size_t shmem = ((size_t)m * LD + (size_t)kb * kb + 2 * kb + 32) * sizeof(float);
    cudaFuncSetAttribute(panel_geqrt_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_geqrt_v2_kernel<<<batch, 256, shmem>>>(
        H.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), V.data_ptr<float>(),
        n, p0, kb, ldt, ldv);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "panel launch failed");
}

/* Whole blocked QR in C++: panel kernel + at::matmul WY trailing, persistent workspace. */
std::vector<torch::Tensor> qr_blocked(torch::Tensor A, int64_t NB) {
    auto H = A.contiguous().clone();
    int64_t batch = H.size(0), n = H.size(1);
    auto opt = H.options();
    auto tau = torch::zeros({batch, n}, opt);
    auto T = torch::empty({batch, NB, NB}, opt);
    auto V = torch::zeros({batch, n, NB}, opt);
    for (int64_t p0 = 0; p0 < n; ) {
        int64_t kb = std::min(NB, n - p0), pe = p0 + kb;
        launch_panel(H, T, tau, V, (int)p0, (int)kb);
        if (pe < n) {
            auto Vt = V.narrow(1, p0, n - p0).narrow(2, 0, kb);          /* (b, m, kb) */
            auto C  = H.narrow(1, p0, n - p0).narrow(2, pe, n - pe);     /* (b, m, n-pe) view */
            auto W  = at::matmul(Vt.transpose(1, 2), C);                 /* Vᵀ C */
            auto Tk = T.narrow(1, 0, kb).narrow(2, 0, kb);
            W = at::matmul(Tk.transpose(1, 2), W);                       /* Tᵀ W */
            C.sub_(at::matmul(Vt, W));                                   /* C -= V W (in place) */
        }
        p0 = pe;
    }
    return {H, tau};
}
"""
CPP_SRC = "std::vector<torch::Tensor> qr_blocked(torch::Tensor A, int64_t NB);"

Kf = load_inline(name="kqr_fused_v2", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                 functions=["qr_blocked"], extra_cuda_cflags=["-O3"], verbose=False)

spec = importlib.util.spec_from_file_location("legacy", QRPY / "householder" / "submission_full.py")
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)

SMEM_FLOATS = 58368        # 228 KB / 4 — B200 SM100 dynamic-SMEM opt-in ceiling


def _fit_nb(n, NB):
    nb = min(NB, n)
    while nb > 8 and n * (nb + 1) + nb * nb + 2 * nb + 32 > SMEM_FLOATS:
        nb -= 8
    return nb


def custom_kernel(A, NB=32):
    return Kf.qr_blocked(A.contiguous(), _fit_nb(A.shape[-1], NB))


STRESS = ["dense", "mixed", "rankdef", "clustered", "nearrank", "nearcollinear",
          "rowscaled", "banded", "uppertri"]


def validate(NB=32, vb=48):
    print(f"\n== FUSED validation (NB={NB}, batch={vb}) ==")
    worst = 0.0
    for n in (176, 352, 512, 1024):
        bad = []
        for case in STRESS:
            A = make_batch(vb, n, 2, case, seed=0).to("cuda").contiguous()
            H, tau = custom_kernel(A, NB)
            fr, og, ft, ot, ps = check(A, H, tau)
            worst = max(worst, fr / ft, og / ot)
            if not ps.all(): bad.append(f"{case}({int(ps.sum())}/{vb})")
        print(f"  n={n:<5d}  {'OK' if not bad else 'FAIL: ' + ','.join(bad)}")
    print(f"  worst margin = {worst:.3f}  (<1.0 == pass)")
    return worst


def _evt(): return torch.cuda.Event(enable_timing=True)


def _time(fn, A, reps=20):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def timing(NB=32):
    print(f"\n== FUSED timing (NB={NB}) vs legacy full ==")
    print(f"{'shape':14s} {'eff':>3s} | {'fused':>8s} {'legacy':>8s} {'speedup':>8s}")
    for n, b in [(176, 40), (352, 40), (512, 640), (1024, 60)]:
        A = make_batch(b, n, 2, "dense", seed=0).to("cuda").contiguous()
        f = _time(lambda x: custom_kernel(x, NB), A)
        leg = _time(legacy.custom_kernel, A)
        print(f"n={n:<5d}b={b:<5d} {_fit_nb(n, NB):>3d} | {f:7.3f}m {leg:7.3f}m {leg/f:7.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    if validate(NB=32) < 1.0:
        timing(NB=16); timing(NB=32); timing(NB=64)
