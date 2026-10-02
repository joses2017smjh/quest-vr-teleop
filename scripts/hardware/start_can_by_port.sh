#!/usr/bin/env bash
# Bring up the four USB-CAN adapters under the interface names the official joint ID mapping uses,
# assigned by USB port instead of kernel probe order (probe order swapped the legs across reboots):
#   can0 = left arm, can1 = right arm, can2 = left leg, can3 = right leg
# Replaces source/berkeley_humanoid_lite_lowlevel/scripts/start_can_transports.sh on this robot.
#
#   sudo scripts/hardware/start_can_by_port.sh            # rename + bring up at 1 Mbit/s
#   scripts/hardware/start_can_by_port.sh --dry-run       # print the commands only (no root needed)
set -euo pipefail

# USB port -> interface name. Limb IDs were verified by ping on 2026-09-12; left/right follow the docs
# (robot's own perspective), which put odd leg IDs on the left leg.
declare -A WANT=(
  [3-4.1:1.0]=can0  # left arm:  1 3 5 7 9
  [3-4.2:1.0]=can1  # right arm: 2 4 6 8 12 (docs expect 10 instead of 12)
  [3-4.4:1.0]=can2  # left leg:  1 3 5 7 11 13
  [3-4.3:1.0]=can3  # right leg: 2 4 6 8 12 14
)
ORDER=(3-4.1:1.0 3-4.2:1.0 3-4.4:1.0 3-4.3:1.0)

DRY_RUN=0
[[ ${1:-} == --dry-run ]] && DRY_RUN=1
run() { if ((DRY_RUN)); then echo "+ $*"; else "$@"; fi; }

declare -A CURRENT=()
for dev in /sys/class/net/*; do
  [[ -e $dev/device ]] || continue
  port=$(basename "$(readlink -f "$dev/device")")
  [[ -n ${WANT[$port]:-} ]] && CURRENT[$port]=$(basename "$dev")
done
for port in "${ORDER[@]}"; do
  [[ -n ${CURRENT[$port]:-} ]] || { echo "No CAN adapter on USB port $port; refusing to guess." >&2; exit 1; }
done

for port in "${ORDER[@]}"; do run ip link set "${CURRENT[$port]}" down; done

# Rename through temporary names so swapping two adapters never collides with a name still in use.
for port in "${ORDER[@]}"; do
  [[ ${CURRENT[$port]} == "${WANT[$port]}" ]] && continue
  run ip link set "${CURRENT[$port]}" name "tmp${WANT[$port]}"
  CURRENT[$port]=tmp${WANT[$port]}
done
for port in "${ORDER[@]}"; do
  [[ ${CURRENT[$port]} == "${WANT[$port]}" ]] && continue
  run ip link set "${CURRENT[$port]}" name "${WANT[$port]}"
done

for port in "${ORDER[@]}"; do run ip link set "${WANT[$port]}" up type can bitrate 1000000; done

((DRY_RUN)) && exit 0
for port in "${ORDER[@]}"; do
  name=${WANT[$port]}
  actual=$(basename "$(readlink -f "/sys/class/net/$name/device")")
  echo "$name -> USB $actual ($(cat "/sys/class/net/$name/operstate"))"
  [[ $actual == "$port" ]] || { echo "MISMATCH: $name is on $actual, expected $port" >&2; exit 1; }
done
