#!/usr/bin/env bash
# profile_nsight.sh — "wall hack" the panel_geqrt kernel on the B200 (nsys timeline).
#
#   ssh b200 'cd /workspace/linalg/take2/tests && bash profile_nsight.sh'
#   ./sync.sh --pull          # fetch results/nsys_*_kernsum.txt
#
# RunPod realities (learned 2026-06-26, encoded here so reconnect is one command):
#  * nsys is NOT a standalone install — use the one bundled with nsight-compute,
#    plus its QdstrmImporter for the .qdstrm -> .nsys-rep step.
#  * QdstrmImporter needs libdw.so.1  -> one-time `apt-get install -y libdw1`.
#  * torch load_inline needs ninja on PATH -> one-time `pip install ninja` + source env.sh
#    (env.sh puts /opt/conda/bin on PATH so torch's `ninja` subprocess resolves).
#  * ncu is PERMS-BLOCKED in this container (ERR_NVGPUCTRPERM) and cannot be fixed on a
#    running pod -> we run nsys only. (CUDA-event splits live in perf_layer4_b200.py.)
set -uo pipefail
cd "$(dirname "$0")"
source /workspace/qr/env.sh 2>/dev/null || true        # /opt/conda/bin (python+ninja) + cuda
PY="${PYTHON:-/opt/conda/bin/python}"
OUT=results; mkdir -p "$OUT"
TS="$(date +%Y%m%d_%H%M%S)"

NSYS="$(command -v nsys || echo /opt/nvidia/nsight-compute/2025.3.0/host/target-linux-x64/nsys)"
IMP="$(ls /opt/nvidia/nsight-compute/*/host/linux-desktop-glibc_*/QdstrmImporter 2>/dev/null | head -1)"

echo "== one-time deps (idempotent) =="
ldconfig -p 2>/dev/null | grep -q 'libdw.so.1' || { apt-get update -qq && apt-get install -y libdw1 >/dev/null 2>&1; }
command -v ninja >/dev/null 2>&1 || "$PY" -m pip install -q ninja
echo "  nsys: $NSYS"; echo "  importer: $IMP"; echo "  ninja: $(command -v ninja || echo MISSING)"
echo

profile() {           # n batch
  local n=$1 b=$2; local base="$OUT/nsys_n${n}_b${b}_${TS}"
  echo "== nsys profile n=$n b=$b =="
  "$NSYS" profile -o "$base" --force-overwrite true -t cuda,nvtx \
    "$PY" profile_driver.py --n "$n" --batch "$b" --iters 20 2>&1 | grep -iE 'done|generat|error' | head
  # newer nsys auto-writes .nsys-rep; if only .qdstrm exists, convert it.
  [ -f "${base}.nsys-rep" ] || { [ -n "$IMP" ] && "$IMP" --input-file "${base}.qdstrm" >/dev/null 2>&1; }
  echo "  -- kernel time ranking (panel vs trailing GEMM vs elementwise) --"
  "$NSYS" stats --report cuda_gpu_kern_sum --format table "${base}.nsys-rep" 2>/dev/null \
    | tee "${base}_kernsum.txt" | head -12
  echo
}

profile 512 640
profile 1024 60

echo "PROFILES IN $OUT/  (pull with ./sync.sh --pull)"
echo "  result so far: panel_geqrt_kernel ~76-78% of GPU time; trailing GEMMs are simt"
echo "  (CUDA-core fp32, ~13%); the panel is the wall -> EG-recursive dgeqrt3 is the lever."
echo "  ncu (SOL/occupancy/TC-util) is perms-blocked on RunPod; nsys timeline only."
