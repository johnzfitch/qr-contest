import sys, pathlib, importlib.util, torch
HERE=pathlib.Path(__file__).resolve(); QRPY=HERE.parents[1]
sys.path[:0]=[str(QRPY),str(QRPY/"common")]
from oracle import make_batch, check
v3=importlib.util.module_from_spec(importlib.util.spec_from_file_location("v3",QRPY/"householder"/"kqr_fused_v3.py"))
importlib.util.spec_from_file_location("v3",QRPY/"householder"/"kqr_fused_v3.py").loader.exec_module(v3)
def _t(fn,A,r=30):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn(A)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/r
for n,bb in [(512,640),(1024,60)]:
    A=make_batch(bb,n,2,"dense",seed=0).cuda().contiguous()
    print(f"n={n}:")
    for nb in (32,48,64):
        H,tau=v3.custom_kernel(A,nb); fr,og,ft,ot,ps=check(A,H,tau)
        t=_t(lambda x: v3.custom_kernel(x,nb),A)
        print(f"  nb={nb:<3d} {t:7.3f}ms  pass={int(ps.sum())}/{bb} margin={max(fr/ft,og/ot):.4f}")
