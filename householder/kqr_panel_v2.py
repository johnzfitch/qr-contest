"""kqr_panel_v2 — PHASE 1: warp-parallel FP32 Householder leaf (path A).

Replaces the scalar one-thread-per-column apply with a warp-per-column,
lane-parallel-over-rows apply. Same (B,T,tau,p0,nb) signature and geqrf-compatible
output convention as the legacy panel_geqrt, so it drops into _hh_qr for validation.

Per reflector j (strict index order; parallelism is WITHIN a step, never across j):
  * warp 0 computes xnorm via a max-scaled sum-of-squares (overflow/underflow-safe,
    LASSQ-equivalent), builds the LAPACK reflector (copysignf/hypotf, tau==0 branch,
    never 1/tau), writes beta into the diagonal, and scales its own column tail.
  * one CTA barrier publishes v + tau.
  * all 8 warps apply H_j to the remaining panel columns (warp per column c, lanes
    over rows r, warp-shuffle reduction for vᵀa), then a -= tau·v·(vᵀa).
  * one CTA barrier completes the update. => ~2 CTA barriers / reflector (path A).
Padded SMEM stride LD=nb+1 kills the stride-32 bank conflict.

  source /workspace/qr/env.sh && python householder/kqr_panel_v2.py
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

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* One CTA (256 threads = 8 warps) per matrix. Factor panel B[p0:n, p0:pe]
   (m=n-p0 rows, nb cols) by warp-parallel Householder, then DLARFT T. */
__global__ void panel_geqrt_v2_kernel(float* __restrict__ B, float* __restrict__ Tg,
                                      float* __restrict__ taug, int n, int p0, int nb) {
    extern __shared__ float smem[];
    int m = n - p0;
    int LD = nb + 1;                       /* padded stride: kills 32-way conflict */
    float* sP   = smem;                    /* m*LD  panel (row-major, padded)      */
    float* sT   = sP + (size_t)m * LD;     /* nb*nb T                              */
    float* stau = sT + (size_t)nb * nb;    /* nb    tau                            */
    float* sz   = stau + nb;               /* nb    DLARFT workspace               */
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int warp = tid >> 5, lane = tid & 31, nwarp = nt >> 5;
    float* Bb = B + (size_t)b * n * n;
    __shared__ float s_tau, s_denom;

    for (int idx = tid; idx < m * nb; idx += nt) {          /* load panel (padded) */
        int r = idx / nb, c = idx % nb;
        sP[(size_t)r * LD + c] = Bb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < nb * nb; idx += nt) sT[idx] = 0.f;
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (warp == 0) {                                   /* --- reflector j (path A) --- */
            float amax = 0.f;                              /* max-scaled ssq norm of tail  */
            for (int r = j + 1 + lane; r < m; r += 32)
                amax = fmaxf(amax, fabsf(sP[(size_t)r * LD + j]));
            amax = warp_max(amax);
            float ssq = 0.f;
            if (amax > 0.f)
                for (int r = j + 1 + lane; r < m; r += 32) {
                    float t = sP[(size_t)r * LD + j] / amax; ssq += t * t;
                }
            ssq = warp_sum(ssq);
            float xnorm = (amax > 0.f) ? amax * sqrtf(ssq) : 0.f;
            if (lane == 0) {
                float alpha = sP[(size_t)j * LD + j];
                float beta, tauj, denom;
                if (xnorm == 0.f) { beta = alpha; tauj = 0.f; denom = 1.f; }   /* identity */
                else {
                    beta  = -copysignf(hypotf(alpha, xnorm), alpha);
                    tauj  = (beta - alpha) / beta;          /* never 1/tau */
                    denom = alpha - beta;
                }
                sP[(size_t)j * LD + j] = beta; stau[j] = tauj; s_tau = tauj; s_denom = denom;
            }
            __syncwarp();
            float denom = s_denom;                          /* warp 0 scales tail: v=tail/denom */
            for (int r = j + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + j] /= denom;
        }
        __syncthreads();                                    /* publish v + tau */

        float tauj = s_tau;
        for (int c = j + 1 + warp; c < nb; c += nwarp) {    /* warp per trailing column */
            float partial = 0.f;
            for (int r = j + 1 + lane; r < m; r += 32)
                partial += sP[(size_t)r * LD + j] * sP[(size_t)r * LD + c];
            float w = (warp_sum(partial) + sP[(size_t)j * LD + c]) * tauj;   /* v[j]=1 term */
            if (lane == 0) sP[(size_t)j * LD + c] -= w;
            for (int r = j + 1 + lane; r < m; r += 32)
                sP[(size_t)r * LD + c] -= sP[(size_t)r * LD + j] * w;
        }
        __syncthreads();                                    /* update complete */
    }

    /* ---- DLARFT: compact-WY T (nb x nb upper-tri), forward columnwise ---- */
    for (int j = 0; j < nb; ++j) {
        if (j > 0) {
            for (int i = tid; i < j; i += nt) {            /* z[i] = -tau_j * (V[:,i].V[:,j]) */
                float d = sP[(size_t)j * LD + i];
                for (int r = j + 1; r < m; ++r) d += sP[(size_t)r * LD + i] * sP[(size_t)r * LD + j];
                sz[i] = -stau[j] * d;
            }
            __syncthreads();
            for (int i = tid; i < j; i += nt) {            /* T[0:j,j] = T[0:j,0:j] @ z[0:j] */
                float acc = 0.f;
                for (int k = i; k < j; ++k) acc += sT[(size_t)i * nb + k] * sz[k];
                sT[(size_t)i * nb + j] = acc;
            }
            __syncthreads();
        }
        if (tid == 0) sT[(size_t)j * nb + j] = stau[j];
        __syncthreads();
    }

    for (int idx = tid; idx < m * nb; idx += nt) {          /* write panel back */
        int r = idx / nb, c = idx % nb;
        Bb[(size_t)(p0 + r) * n + (p0 + c)] = sP[(size_t)r * LD + c];
    }
    for (int idx = tid; idx < nb * nb; idx += nt) Tg[(size_t)b * nb * nb + idx] = sT[idx];
    for (int i = tid; i < nb; i += nt) taug[(size_t)b * n + (p0 + i)] = stau[i];
}

void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb) {
    int batch = B.size(0), n = B.size(1);
    int m = n - p0, LD = nb + 1;
    size_t shmem = ((size_t)m * LD + (size_t)nb * nb + 2 * nb + 32) * sizeof(float);
    cudaFuncSetAttribute(panel_geqrt_v2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_geqrt_v2_kernel<<<batch, 256, shmem>>>(
        B.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), n, p0, nb);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
"""
CPP_SRC = "void panel_geqrt(torch::Tensor B, torch::Tensor T, torch::Tensor tau, int p0, int nb);"

Kv2 = load_inline(name="kqr_panel_v2", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                  functions=["panel_geqrt"], extra_cuda_cflags=["-O3"], verbose=False)

# --- legacy control (the proven scalar kernel) for cross-check ---
spec = importlib.util.spec_from_file_location("legacy", QRPY / "householder" / "submission_full.py")
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)


SMEM_FLOATS = 56 * 1024            # ~224KB / 4, safe under the B200 dynamic-SMEM cap


def _fit_nb(n, NB):
    """Largest nb<=NB whose first-panel SMEM (m=n, padded LD) fits the cap."""
    nb = min(NB, n)
    while nb > 8 and n * (nb + 1) + nb * nb + 2 * nb + 32 > SMEM_FLOATS:
        nb -= 8
    return nb


def _hh_qr_v2(A, NB):
    b, n, _ = A.shape
    B = A.contiguous().clone()
    tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    NBf = _fit_nb(n, NB)
    p0 = 0
    while p0 < n:
        nb = min(NBf, n - p0); pe = p0 + nb
        T = torch.zeros(b, nb, nb, device=A.device, dtype=A.dtype)
        Kv2.panel_geqrt(B, T, tau, p0, nb)
        if pe < n:
            V = torch.tril(B[:, p0:, p0:pe], -1); V[:, :nb, :nb] += eye[:nb, :nb]
            C = B[:, p0:, pe:]
            VtC = torch.einsum('bmi,bmc->bic', V, C)
            TtVtC = torch.einsum('bki,bkc->bic', T, VtC)
            B[:, p0:, pe:] = C - torch.einsum('bmi,bic->bmc', V, TtVtC)
        p0 = pe
    return B, tau


STRESS = ["dense", "mixed", "rankdef", "clustered", "nearrank", "nearcollinear",
          "rowscaled", "banded", "uppertri"]


def validate(NB=64, vb=48):
    print(f"\n== VALIDATION (NB={NB}, batch={vb})  worst factor/ortho margin per case ==")
    print(f"{'n':>5s} {'case':13s} {'fac/tol':>9s} {'orth/tol':>9s} {'pass':>7s}  legacy")
    worst = 0.0
    for n in (176, 352, 512, 1024):
        for case in STRESS:
            A = make_batch(vb, n, 2, case, seed=0).to("cuda").contiguous()
            H, tau = _hh_qr_v2(A, NB)
            fr, og, ft, ot, ps = check(A, H, tau)
            Hl, tl = legacy.custom_kernel(A)
            _, _, _, _, psl = check(A, Hl, tl)
            fm, om = fr / ft, og / ot
            worst = max(worst, fm, om)
            flag = "OK" if ps.all() else "**FAIL**"
            print(f"{n:>5d} {case:13s} {fm:9.3f} {om:9.3f} {f'{int(ps.sum())}/{vb}':>7s}  "
                  f"{'legOK' if psl.all() else 'legFAIL'} {flag if not ps.all() else ''}")
    print(f"  worst margin across all = {worst:.3f}  (<1.0 == pass)")
    return worst


def _evt():
    return torch.cuda.Event(enable_timing=True)


def _panel_only_v2(A, NB, reps=20):
    b, n, _ = A.shape
    B = A.contiguous().clone(); tau = torch.zeros(b, n, device=A.device, dtype=A.dtype)
    NBf = _fit_nb(n, NB)
    def one():
        p0 = 0
        while p0 < n:
            nb = min(NBf, n - p0)
            T = torch.empty(b, nb, nb, device=A.device, dtype=A.dtype)
            Kv2.panel_geqrt(B, T, tau, p0, nb); p0 += nb
    for _ in range(3): one()
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): one()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def _full_v2(A, NB, reps=20):
    for _ in range(3): _hh_qr_v2(A, NB)
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): _hh_qr_v2(A, NB)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def _legacy_full(A, reps=20):
    for _ in range(3): legacy.custom_kernel(A)
    torch.cuda.synchronize(); s, e = _evt(), _evt(); s.record()
    for _ in range(reps): legacy.custom_kernel(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def timing():
    SHAPES = [(176, 40), (352, 40), (512, 640), (1024, 60)]
    print(f"\n== TIMING: v2 panel-only + full vs legacy full (nb sweep) ==")
    print(f"{'shape':13s} {'nb':>3s} {'eff':>3s} | {'v2 panel':>9s} {'v2 full':>9s} "
          f"{'legacy':>8s} {'full spd':>8s}")
    for n, b in SHAPES:
        A = make_batch(b, n, 2, "dense", seed=0).to("cuda").contiguous()
        leg = _legacy_full(A)
        for NB in (16, 32, 64):
            eff = _fit_nb(n, NB)
            p = _panel_only_v2(A, NB); f = _full_v2(A, NB)
            print(f"n={n:<5d}b={b:<4d} {NB:>3d} {eff:>3d} | {p:8.3f}m {f:8.3f}m "
                  f"{leg:7.3f}m {leg/f:7.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    w = validate(NB=64)
    if w < 1.0:
        timing()
