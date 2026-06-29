import sys, pathlib, importlib.util, math, torch
HERE=pathlib.Path(__file__).resolve(); QRPY=HERE.parents[1]
sys.path[:0]=[str(QRPY),str(QRPY/"common")]
from oracle import make_batch, SHAPES
s=importlib.util.spec_from_file_location("sub",QRPY/"householder"/"jerry_owes_me_lunch.py")
sub=importlib.util.module_from_spec(s); s.loader.exec_module(sub)
def _t(fn,A,r=30):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn(A)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/r
logs=[]
print(f"{'shape':22s} {'ms':>8s}")
for (b,n,cond,case) in SHAPES:
    A=make_batch(b,n,cond,case,seed=0).cuda().contiguous()
    t=_t(lambda x: sub.custom_kernel(x),A); logs.append(math.log(t))
    print(f"b{b:<4d}n{n:<5d}{case:9s} {t:8.3f}")
gm=math.exp(sum(logs)/len(logs))
print(f"\n12-shape GEOMEAN = {gm:.3f} ms   (n={len(logs)} instances)")
