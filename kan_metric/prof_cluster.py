"""Profile where the cluster KAN kernel spends time (b640 n512 C16)."""
import torch
# --- path shim (reorg): reach common/ (oracle) + root (task) ---
import sys, pathlib
_qr = pathlib.Path(__file__).resolve().parents[1]      # linalg/qr_py
sys.path[:0] = [str(_qr / 'common'), str(_qr)]
# --- end shim ---
from kqr_cluster_kan import _build, _bench
from oracle import make_batch

A = make_batch(640, 512, 2, "dense", seed=0)
prev = 0.0
labels = {1: "gram", 2: "+chol", 3: "+trsm (1 CQR pass)", 4: "+2nd pass (CQR2)"}
for stage in (1, 2, 3, 4):
    m = _build(stage)
    ms = _bench(lambda: m.cluster_kan(A, 16))
    print(f"  STAGE{stage} {labels[stage]:22s} total={ms:7.1f} ms   delta={ms-prev:7.1f} ms")
    prev = ms
mf = _build(4)
msf = _bench(lambda: mf.cluster_qr(A, 16))
print(f"  full cluster_qr (CQR2+modLU+closure) total={msf:7.1f} ms   "
      f"modLU+closure ~= {msf-prev:7.1f} ms")
