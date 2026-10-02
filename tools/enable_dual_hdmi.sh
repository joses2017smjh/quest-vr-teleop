#!/usr/bin/env bash
# Two HDMI screens (the PC monitor + the robot's face display) on this Beelink EQ.
#
# Why only one works now: the N150's graphics (PCI 8086:46D4) are unknown to Ubuntu
# 22.04's kernel (6.8). Intel support for this chip starts in Linux 6.9, so today the
# desktop runs on the firmware's framebuffer (simpledrm): ONE screen, fixed resolution,
# software rendering (llvmpipe). The i915.force_probe=46d4 already on the boot line
# cannot help: force_probe only works for chips in the driver's list, and 46D4 is not.
#
# What this does: installs Ubuntu's mainline build of the 6.12 LTS kernel NEXT TO the
# current one. Nothing is removed. The robot's drivers (CAN gs_usb, camera uvcvideo,
# serial ch341/cp210x, Wi-Fi iwlwifi) are all standard and come with it, and there are
# no DKMS modules to rebuild. Secure Boot is off, so the unsigned image boots.
#
#   tools/enable_dual_hdmi.sh           download, verify checksums, install (asks for sudo)
#   tools/enable_dual_hdmi.sh --check   is the Intel driver running? which screens are seen?
#   tools/enable_dual_hdmi.sh --download   only fetch and verify (no sudo); the install is then quick
#
# Then reboot. The boot menu shows for 10 s; the new kernel is the default.
# If anything misbehaves: at the boot menu choose "Advanced options for Ubuntu" ->
# "Ubuntu, with Linux 6.8.0-138-generic". To remove it for good:
#   sudo apt remove linux-image-unsigned-6.12.111-0612111-generic linux-modules-6.12.111-0612111-generic
#
# Caveat: mainline builds get no automatic security updates from apt.

set -euo pipefail

VER=6.12.111
BUILD=0612111
STAMP=202609211340
BASE="https://kernel.ubuntu.com/mainline/v${VER}/amd64"
IMAGE="linux-image-unsigned-${VER}-${BUILD}-generic_${VER}-${BUILD}.${STAMP}_amd64.deb"
MODULES="linux-modules-${VER}-${BUILD}-generic_${VER}-${BUILD}.${STAMP}_amd64.deb"
DIR="${HOME}/.cache/bhl/kernel-${VER}"

check() {
    echo "kernel:        $(uname -r)"
    local driver
    driver=$(basename "$(readlink -f /sys/class/drm/card*/device/driver 2>/dev/null | head -1)" 2>/dev/null || true)
    echo "GPU driver:    ${driver:-none}  ($(awk '{print $2}' /proc/fb 2>/dev/null | head -1))"
    echo "screens seen:"
    for d in /sys/class/drm/card*-*; do
        [ -e "$d/status" ] || continue
        printf "  %-26s %-13s %s\n" "$(basename "$d")" "$(cat "$d/status")" "$(head -1 "$d/modes" 2>/dev/null)"
    done
    if command -v glxinfo >/dev/null; then
        glxinfo -B 2>/dev/null | grep -E "OpenGL renderer|Accelerated" | sed 's/^ */  /'
    fi
    if [ "${driver:-}" = "i915" ]; then
        echo "=> The Intel driver is running: both HDMI ports can be used. Settings -> Displays to arrange them."
    else
        echo "=> Still on the firmware framebuffer: one screen only. Run this script without --check, then reboot."
    fi
}

if [ "${1:-}" = "--check" ]; then
    check
    exit 0
fi

mkdir -p "$DIR"
cd "$DIR"
echo "Downloading Linux ${VER} (about 200 MB) to ${DIR} ..."
for f in "$IMAGE" "$MODULES" CHECKSUMS; do
    [ -s "$f" ] || curl -fSL --retry 3 -o "$f" "${BASE}/${f}"
done
echo "Verifying checksums ..."
grep -E "^[0-9a-f]{64}  (${IMAGE}|${MODULES})$" CHECKSUMS | sha256sum -c -
if [ "${1:-}" = "--download" ]; then
    echo "Downloaded and verified. To install: tools/enable_dual_hdmi.sh   (then reboot)"
    exit 0
fi
echo "Installing next to the current kernel (sudo asks for your password) ..."
sudo apt-get install -y "./${IMAGE}" "./${MODULES}"
echo
echo "Installed. Reboot when the robot is stopped: sudo reboot"
echo "After the reboot, check with: tools/enable_dual_hdmi.sh --check"
