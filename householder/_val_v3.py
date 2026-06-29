import sys, pathlib, importlib.util, torch
HERE=pathlib.Path(__file__).resolve(); QRPY=HERE.parents[1]
sys.path[:0]=[str(QRPY),str(QRPY/"common")]
from oracle import make_batch, check, SHAPES
s=importlib.util.spec_from_file_location("sub",QRPY/"householder"/"submission_routed_v3.py")
sub=importlib.util.module_from_spec(s); s.loader.exec_module(sub)
worst=0.0; allok=True
for (b,n,cond,case) in SHAPES:
    A=make_batch(b,n,cond,case,seed=0).cuda().contiguous()
    H,tau=sub.custom_kernel(A); fr,og,ft,ot,ps=check(A,H,tau)
    m=max(fr/ft,og/ot); worst=max(worst,m); ok=bool(ps.all()); allok&=ok
    print(f"  b{b:<4d}n{n:<5d}{case:11s} pass={int(ps.sum())}/{b} margin={m:.4f} {'OK' if ok else 'FAIL'}")
print(f"\nworst margin={worst:.4f}  {'ALL PASS' if allok else 'SOME FAIL'}")
