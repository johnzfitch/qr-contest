"""kqr_wavefront_v3 — OPTION A: intra-CTA column-warp WAVEFRONT panel.

Same outer scaffolding as kqr_fused_v2 (probe #7): qr_blocked runs the whole
blocked loop in C++, the panel kernel writes compact (H,tau)+T+explicit-V into a
persistent workspace, trailing is at::matmul WY. ONLY the panel kernel body changes.

The phase-1/-2 panel kernel (panel_geqrt_v2) is RIGHT-LOOKING: a serial j-loop with
TWO __syncthreads per reflector (full-CTA barriers in the inner loop). This kernel
replaces that with a column-warp DAG:

  * ONE warp per column (nwarp = kb, block = kb*32, <=1024). Warp c OWNS column c.
  * Warp c applies reflectors 0..c-1 to its own column as each becomes ready
    (acquire-load on ready[k]), then GENERATES reflector c and publishes it
    (release-store on ready[c]).  NO full-CTA barrier in the reflector loop — only
    per-column ready flags + __syncwarp.  Warp c+1's applies overlap warp c's generate.
  * V-publication invariant (load-bearing): all 32 lanes __syncwarp(FULL) after
    writing the v_c tail + beta, then __threadfence_block(), THEN lane0 release-stores
    ready[c].  Consumers acquire-load ready[k] before touching column k.  This is the
    fix for the stale-V footgun (release ordering the tail but not the V tile).
  * cuda::atomic_ref<int, thread_scope_block> for the flags — name-safe + runtime
    confirmed (nmprobe nm4). Single __syncthreads() at the panel/T boundary only.

Ships clean via bare <<<>>> exactly like probe #7. Becomes the panel-worker for B.

  source /workspace/qr/env.sh && python householder/kqr_wavefront_v3.py
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
#include <cuda/atomic>
#include <vector>
#include <algorithm>

#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

/* One CTA per matrix, ONE WARP per column (nwarp == kb). Factor panel
   H[p0:n, p0:p0+kb] as a column DAG, write compact (H,tau)+T and the EXPLICIT V tile
   (diag=1, above=0, below=tail) into Vout[:, p0:n, 0:kb]. */
__global__ void panel_wavefront_v3_kernel(float* __restrict__ Hm, float* __restrict__ Tg,
                                          float* __restrict__ taug, float* __restrict__ Vout,
                                          int n, int p0, int kb, int ldt, int ldv) {
    extern __shared__ float smem[];
    int m = n - p0, LD = kb + 1;
    float* sP   = smem;                       /* m  x LD   panel (padded)         */
    float* sT   = sP + (size_t)m * LD;        /* kb x kb   T                       */
    float* stau = sT + (size_t)kb * kb;       /* kb        tau                     */
    float* sz   = stau + kb;                  /* kb        DLARFT scratch          */
    int*   ready = (int*)(sz + kb);           /* kb        column-ready epochs     */
    int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int warp = tid >> 5, lane = tid & 31;
    float* Hb = Hm + (size_t)b * n * n;

    /* load panel, zero T, clear ready flags */
    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, c = idx % kb;
        sP[(size_t)r * LD + c] = Hb[(size_t)(p0 + r) * n + (p0 + c)];
    }
    for (int idx = tid; idx < kb * kb; idx += nt) sT[idx] = 0.f;
    for (int idx = tid; idx < kb;      idx += nt) ready[idx] = 0;
    __syncthreads();

    /* ---- WAVEFRONT: warp c owns column c ---- */
    int c = warp;                              /* nwarp == kb, so c in [0, kb) */
    if (c < kb) {
        /* apply reflectors 0..c-1 to column c as they become ready */
        for (int k = 0; k < c; ++k) {
            cuda::atomic_ref<int, cuda::thread_scope_block> fr(ready[k]);
            while (fr.load(cuda::memory_order_acquire) == 0) { __nanosleep(32); }
            float tauk = stau[k];
            float partial = 0.f;
            for (int r = k + 1 + lane; r < m; r += 32)
                partial += sP[(size_t)r * LD + k] * sP[(size_t)r * LD + c];
            float w = (warp_sum(partial) + sP[(size_t)k * LD + c]) * tauk;
            if (lane == 0) sP[(size_t)k * LD + c] -= w;
            for (int r = k + 1 + lane; r < m; r += 32)
                sP[(size_t)r * LD + c] -= sP[(size_t)r * LD + k] * w;
            __syncwarp(FULL);
        }
        /* generate reflector c from column c (pivot row c, tail rows c+1..m-1) */
        float amax = 0.f;
        for (int r = c + 1 + lane; r < m; r += 32) amax = fmaxf(amax, fabsf(sP[(size_t)r * LD + c]));
        amax = warp_max(amax);
        float ssq = 0.f;
        if (amax > 0.f) for (int r = c + 1 + lane; r < m; r += 32) { float t = sP[(size_t)r * LD + c] / amax; ssq += t * t; }
        ssq = warp_sum(ssq);
        float xnorm = (amax > 0.f) ? amax * sqrtf(ssq) : 0.f;
        float denom;
        if (lane == 0) {
            float alpha = sP[(size_t)c * LD + c], beta, tauc;
            if (xnorm == 0.f) { beta = alpha; tauc = 0.f; denom = 1.f; }
            else { beta = -copysignf(hypotf(alpha, xnorm), alpha); tauc = (beta - alpha) / beta; denom = alpha - beta; }
            sP[(size_t)c * LD + c] = beta; stau[c] = tauc;
        }
        denom = __shfl_sync(FULL, denom, 0);
        for (int r = c + 1 + lane; r < m; r += 32) sP[(size_t)r * LD + c] /= denom;
        /* ---- V-publication fence. DO NOT "simplify". The three steps are all load-bearing:
           (1) __syncwarp converges the warp: every lane has ISSUED its v_c-tail store.
           (2) __threadfence_block on EVERY lane orders THAT lane's writes at block scope
               (a single lane-0 release-store would NOT order lanes 1..31's tail writes:
                __syncwarp is a control barrier, not a memory barrier; on Volta+ ITS a
                lane's store can be issued-but-not-yet-visible).
           (3) the second __syncwarp ensures every lane's fence has been issued before
               lane 0's release-store publishes ready[c]. Consumers acquire-load ready[c]
               before reading column c, completing the release/acquire pair. ---- */
        __syncwarp(FULL);
        __threadfence_block();
        __syncwarp(FULL);
        if (lane == 0)
            cuda::atomic_ref<int, cuda::thread_scope_block>(ready[c]).store(1, cuda::memory_order_release);
    }
    __syncthreads();                           /* single panel/T boundary barrier */

    /* ---- DLARFT T (identical to v2) ---- */
    for (int j = 0; j < kb; ++j) {
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

    /* ---- compact H + explicit V + T + tau out (identical to v2) ---- */
    for (int idx = tid; idx < m * kb; idx += nt) {
        int r = idx / kb, cc = idx % kb;
        float val = sP[(size_t)r * LD + cc];
        Hb[(size_t)(p0 + r) * n + (p0 + cc)] = val;
        float vv = (r == cc) ? 1.f : ((r > cc) ? val : 0.f);
        Vout[(size_t)b * n * ldv + (size_t)(p0 + r) * ldv + cc] = vv;
    }
    for (int i = tid; i < kb; i += nt)
        for (int j = 0; j < kb; ++j) Tg[(size_t)b * ldt * ldt + (size_t)i * ldt + j] = sT[(size_t)i * kb + j];
    for (int i = tid; i < kb; i += nt) taug[(size_t)b * n + (p0 + i)] = stau[i];
}

static void launch_panel(torch::Tensor H, torch::Tensor T, torch::Tensor tau,
                         torch::Tensor V, int p0, int kb) {
    int batch = H.size(0), n = H.size(1), m = n - p0, LD = kb + 1;
    int ldt = T.size(2), ldv = V.size(2);
    TORCH_CHECK(kb <= 32, "wavefront_v3 is one-warp-per-column: kb must be <= 32");
    size_t shmem = ((size_t)m * LD + (size_t)kb * kb + 3 * kb + 32) * sizeof(float);
    int block = kb * 32;                       /* one warp per column */
    cudaFuncSetAttribute(panel_wavefront_v3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    panel_wavefront_v3_kernel<<<batch, block, shmem>>>(
        H.data_ptr<float>(), T.data_ptr<float>(), tau.data_ptr<float>(), V.data_ptr<float>(),
        n, p0, kb, ldt, ldv);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "panel launch failed");
}

/* Whole blocked QR in C++: wavefront panel kernel + at::matmul WY trailing. */
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
            auto Vt = V.narrow(1, p0, n - p0).narrow(2, 0, kb);
            auto C  = H.narrow(1, p0, n - p0).narrow(2, pe, n - pe);
            auto W  = at::matmul(Vt.transpose(1, 2), C);
            auto Tk = T.narrow(1, 0, kb).narrow(2, 0, kb);
            W = at::matmul(Tk.transpose(1, 2), W);
            C.sub_(at::matmul(Vt, W));
        }
        p0 = pe;
    }
    return {H, tau};
}
"""
CPP_SRC = "std::vector<torch::Tensor> qr_blocked(torch::Tensor A, int64_t NB);"

Kf = load_inline(name="kqr_wavefront_v3", cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                 functions=["qr_blocked"], extra_cuda_cflags=["-O3"], verbose=False)

spec = importlib.util.spec_from_file_location("legacy", QRPY / "householder" / "submission_full.py")
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)

SMEM_FLOATS = 58368        # 228 KB / 4 — B200 SM100 dynamic-SMEM opt-in ceiling


def _fit_nb(n, NB):
    nb = min(NB, n)
    while nb > 8 and n * (nb + 1) + nb * nb + 3 * nb + 32 > SMEM_FLOATS:
        nb -= 8
    return min(nb, 32)         # one-warp-per-column: kb capped at 32 (1024 threads)


def custom_kernel(A, NB=32):
    return Kf.qr_blocked(A.contiguous(), _fit_nb(A.shape[-1], NB))


STRESS = ["dense", "mixed", "rankdef", "clustered", "nearrank", "nearcollinear",
          "rowscaled", "banded", "uppertri"]


def validate(NB=32, vb=48):
    print(f"\n== WAVEFRONT validation (NB={NB}, batch={vb}) ==")
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
    print(f"\n== WAVEFRONT timing (NB={NB}) vs fused-v2 ==")
    try:
        spec2 = importlib.util.spec_from_file_location("fusedv2", QRPY / "householder" / "kqr_fused_v2.py")
        fv2 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(fv2)
        ref = fv2.custom_kernel
    except Exception:
        ref = legacy.custom_kernel
    print(f"{'shape':14s} {'kb':>3s} | {'wave':>8s} {'ref':>8s} {'speedup':>8s}")
    for n, b in [(176, 40), (352, 40), (512, 640), (1024, 60)]:
        A = make_batch(b, n, 2, "dense", seed=0).to("cuda").contiguous()
        w = _time(lambda x: custom_kernel(x, NB), A)
        r = _time(lambda x: ref(x, NB) if ref is not legacy.custom_kernel else ref(x), A)
        print(f"n={n:<5d}b={b:<5d} {_fit_nb(n, NB):>3d} | {w:7.3f}m {r:7.3f}m {r/w:7.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    if validate(NB=32) < 1.0:
        timing(NB=16); timing(NB=32)
