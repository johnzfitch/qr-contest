import sys, pathlib, importlib.util, torch
HERE=pathlib.Path(__file__).resolve(); QRPY=HERE.parents[1]
sys.path[:0]=[str(QRPY),str(QRPY/"common")]
from oracle import make_batch, check
kb=importlib.util.module_from_spec(importlib.util.spec_from_file_location("kb",QRPY/"kan_metric"/"kqr_blocked.py"))
importlib.util.spec_from_file_location("kb",QRPY/"kan_metric"/"kqr_blocked.py").loader.exec_module(kb)
def _t(fn,A,r=20):
    for _ in range(3): fn(A)
    torch.cuda.synchronize(); a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(r): fn(A)
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b)/r
print("CQR2 pipeline (AtA big GEMM + chol + modLU recon) vs HH-fused, n512/n1024 b-high:")
for n,bb,cond,case in [(512,640,2,"dense"),(512,640,0,"rankdef"),(1024,60,2,"dense")]:
    A=make_batch(bb,n,cond,case,seed=0).cuda().contiguous()
    H,tau=kb.pipeline(A); fr,og,ft,ot,ps=check(A,H,tau)
    t=_t(lambda x: kb.pipeline(x),A)
    # decompose
    Q,Rc,ok=kb.robust_cqr(A,passes=2); tcqr=_t(lambda x: kb.robust_cqr(x,passes=2),A); tmod=_t(lambda x: kb.blocked_modlu(Q),A)
    print(f"  n{n} b{bb} {case:8s}: pipeline={t:6.2f}ms (cqr2={tcqr:.2f} modlu={tmod:.2f}) pass={int(ps.sum())}/{bb} margin={max(fr/ft,og/ot):.3f}")
