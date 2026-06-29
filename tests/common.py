"""Shared harness for the QR-contest test suite (TEST_SUITE_PLAN.md).

Standalone-script style (matches repo idiom: oracle.py / sweep.py / testkit). Each
test_layerN_*.py imports this, builds a Runner, asserts, and exits nonzero on any
hard failure. Accuracy is CPU-representative; only Layer 4 (perf) needs the B200.

The artifact under test is qr_hh_portfolio.py, whose kernel reads the arch
parameters (FMT/SLICES/TERMS/NB/EDGE_GUARD) as MODULE GLOBALS at call time. So we
can exercise any arch in-process by set_arch(n) — no per-arch subprocess needed.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# oracle.py lives in linalg/qr_py/common ; portfolio in linalg/take2/sweeps
sys.path.insert(0, os.path.join(_REPO, "linalg", "qr_py", "common"))
sys.path.insert(0, os.path.join(_REPO, "linalg", "take2", "sweeps"))

import torch                       # noqa: E402
import oracle                      # noqa: E402
import qr_hh_portfolio as port     # noqa: E402

DEV = oracle.DEV
EPS32 = oracle.EPS32

# Keep CPU runs sane; on CUDA we use the full benchmark batch.
CPU_BATCH_CAP = 4
CPU_N_CAP = 1024   # Layer 2 skips larger n on CPU (fp64 blocked_hh is heavy off-GPU)


# --------------------------------------------------------------------------- #
# Arch selection — mutate the portfolio's module globals in place.            #
# --------------------------------------------------------------------------- #
# Grouped by the CPU-verified verdict from chat 2 (see TEST_SUITE_PLAN.md S2).
ARCH_CONTROL = [1, 2, 3]            # fp32 trailing — correctness truth / baseline
ARCH_BF16    = [4, 6, 8]            # bf16 Ozaki families (9t, 6t, x2/3t)
ARCH_FP8     = [10, 12, 13]        # fp8 families (+ edge-guard hybrid)
ARCH_NVFP4   = [14, 15, 16, 17, 18]  # confirmed NEGATIVE — xfail
ARCH_HYBRID  = [19, 20]            # bf16/fp8 + fp32 edges

# Default "should pass" set for the gate layers.
ARCH_PASS = ARCH_CONTROL[:1] + ARCH_BF16 + ARCH_FP8 + ARCH_HYBRID


def set_arch(arch):
    """Point the portfolio's globals at ARCH_TABLE[arch]; return its note string."""
    fmt, slices, terms, nb, edge, note = port.ARCH_TABLE[arch]
    port.FMT = fmt
    port.SLICES = slices
    port.TERMS = terms
    port.NB = nb
    port.EDGE_GUARD = edge
    port.ARCH = arch
    return note


def arch_label(arch):
    fmt, slices, terms, nb, edge, _ = port.ARCH_TABLE[arch]
    return f"a{arch}({fmt}x{slices}/{terms}t,nb{nb}{',edge' if edge else ''})"


# --------------------------------------------------------------------------- #
# Batch / shape caps so the same scripts run on CPU and B200.                  #
# --------------------------------------------------------------------------- #
def cap_batch(b):
    return b if DEV == "cuda" else min(b, CPU_BATCH_CAP)


def skip_on_cpu(n):
    return DEV != "cuda" and n > CPU_N_CAP


# --------------------------------------------------------------------------- #
# Tiny standalone test runner (no pytest dependency).                          #
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, layer):
        self.layer = layer
        self.passed = 0
        self.fails = []
        self.xpass = []   # things that unexpectedly passed an expected-fail
        print(f"=== {layer}  (device={DEV}, eps32={EPS32:.3e}) ===")

    def ok(self, name, cond, detail=""):
        cond = bool(cond)
        tag = "PASS" if cond else "FAIL"
        print(f"  [{tag}] {name}  {detail}")
        if cond:
            self.passed += 1
        else:
            self.fails.append(name)
        return cond

    def xfail(self, name, failed_as_expected, detail=""):
        """Expected-negative: PASS the suite iff the thing FAILED its gate."""
        failed_as_expected = bool(failed_as_expected)
        tag = "XFAIL-OK" if failed_as_expected else "XPASS!!"
        print(f"  [{tag}] {name}  {detail}")
        if failed_as_expected:
            self.passed += 1
        else:
            self.xpass.append(name)
        return failed_as_expected

    def done(self):
        total = self.passed + len(self.fails) + len(self.xpass)
        print(f"\n{self.layer}: {self.passed}/{total} passed")
        if self.fails:
            print("  FAILED: " + ", ".join(self.fails))
        if self.xpass:
            print("  UNEXPECTED PASS (expected-negative now passing): " + ", ".join(self.xpass))
        bad = self.fails or self.xpass
        sys.exit(1 if bad else 0)
