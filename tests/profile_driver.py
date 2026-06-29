"""Profiling driver for the panel_geqrt Householder artifact (nsys / ncu).

Loads householder/submission_full.py (the board-proven ~13ms panel_geqrt kernel,
compiled via load_inline at import) and runs its custom_kernel on a target shape
under an NVTX range, so the profiler attributes the whole pipeline:
  panel_geqrt_kernel  (scalar geqr2, the suspected ~96% wall)
  + the 3 trailing einsums (VtC, TtVtC, V@TtVtC)
  + python-loop / tril/clone/eye orchestration gaps between panels.
That last split (kernel compute vs orchestration) is exactly what the timeline
answers and end-to-end popcorn timing cannot.

  /opt/conda/bin/python profile_driver.py --n 512 --batch 640 --iters 20
Driven by profile_nsight.sh. n<1536 stays on the Householder path (not geqrf).
"""
import argparse, sys, pathlib, importlib.util
import torch

HERE = pathlib.Path(__file__).resolve()
LINALG = HERE.parents[2]                 # tests -> take2 -> linalg
QRPY = LINALG / "qr_py"
sys.path[:0] = [str(QRPY), str(QRPY / "common")]   # for `task` and `oracle`

from oracle import make_batch                       # noqa: E402

ART = QRPY / "householder" / "submission_full.py"
spec = importlib.util.spec_from_file_location("panel_artifact", ART)
art = importlib.util.module_from_spec(spec)
spec.loader.exec_module(art)             # triggers load_inline CUDA compile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=640)
    ap.add_argument("--case", default="dense")
    ap.add_argument("--cond", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "need CUDA"

    A = make_batch(a.batch, a.n, a.cond, a.case, seed=0).to("cuda").contiguous()
    for _ in range(a.warmup):
        art.custom_kernel(A)
    torch.cuda.synchronize()

    tag = f"qr_n{a.n}_b{a.batch}"
    for _ in range(a.iters):
        torch.cuda.nvtx.range_push(tag)
        art.custom_kernel(A)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"done {tag}  case={a.case} iters={a.iters}")


if __name__ == "__main__":
    main()
