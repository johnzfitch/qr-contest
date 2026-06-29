import sys, pathlib, importlib.util
import torch
HERE = pathlib.Path(__file__).resolve(); QRPY = HERE.parents[1]
sys.path[:0] = [str(QRPY), str(QRPY/"common")]
from oracle import make_batch, check
def _imp(p):
    s=importlib.util.spec_from_file_location(p.stem,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v3=_imp(QRPY/"householder"/"kqr_fused_v3.py")
wave=_imp(QRPY/"householder"/"kqr_wavefront_v3.py")
def _t(fn,A,r=20):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(r): fn(A)
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/r
print(f"{'shape':14s} {'v3-fused':>10s} {'wavefront':>10s}  winner")
for n,b in [(176,40),(352,40),(1024,60)]:
    A=make_batch(b,n,2,"dense",seed=0).cuda().contiguous()
    f=_t(lambda x: v3.custom_kernel(x,32),A); w=_t(lambda x: wave.custom_kernel(x,32),A)
    print(f"n={n:<5d}b={b:<5d} {f:9.3f}m {w:9.3f}m  {'v3-fused' if f<w else 'wavefront'}")
