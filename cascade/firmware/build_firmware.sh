#!/bin/bash
# Rebuild am273x_cascade_param.xer5f from src/ in the Lima VM "ti" (TI mmWave MCU+ SDK
# 04.04.00.01, ti-cgt-armllvm 2.1.2.LTS, SysConfig 1.14; x86 tools under qemu-user), see
# README.md. The VM must mount this folder writable.
#   ./build_firmware.sh                        -> am273x_cascade_param.xer5f
#   ./build_firmware.sh other_src other.xer5f
set -euo pipefail
cd "$(dirname "$0")"
SRC="$(cd "${1:-src}" && pwd)"
OUT="$(pwd)/${2:-am273x_cascade_param.xer5f}"
limactl start ti >/dev/null 2>&1 || true
limactl shell ti -- bash -c "
set -e
S=\$HOME/ti/mmwave_mcuplus_sdk_04_04_00_01/mmwave_mcuplus_sdk_04_04_00_01
cp '$SRC'/*.c '$SRC'/*.h \$S/ti/utils/test/cascade/am273x/
cd \$S/scripts/unix && source ./setenv_mac.sh >/dev/null 2>&1 || true
cd \$S/ti/utils/test/cascade && make test 2>&1 | grep -E 'error|warning: |Built'
cp am273x/am273x_cascade_mss.xer5f '$OUT'
cp am273x/am273x_cascade_mss.map '${OUT%.xer5f}.map'
"
ls -la "$OUT"
