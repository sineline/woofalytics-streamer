#!/usr/bin/env bash
# One-time host setup for Manjaro/Arch: installs NVIDIA Container Toolkit via AUR
# and configures Docker to use the nvidia runtime.
# Run with: bash setup_nvidia_docker.sh
set -euo pipefail

# ── 1. Install yay (AUR helper) if not present ────────────────────────────────
if ! command -v yay &>/dev/null; then
    echo "==> yay not found — building from AUR..."
    TMPDIR=$(mktemp -d)
    git clone https://aur.archlinux.org/yay.git "$TMPDIR/yay"
    pushd "$TMPDIR/yay"
    makepkg -si --noconfirm
    popd
    rm -rf "$TMPDIR"
else
    echo "==> yay already installed: $(yay --version)"
fi

# ── 2. Refresh pacman keyring (fixes "unknown trust" signature errors) ────────
echo "==> Refreshing pacman keyring..."
sudo pacman -Sy --noconfirm archlinux-keyring

# ── 3. Install nvidia-container-toolkit ───────────────────────────────────────
echo "==> Installing nvidia-container-toolkit..."
yay -S --noconfirm nvidia-container-toolkit

# ── 3. Configure Docker nvidia runtime ────────────────────────────────────────
echo "==> Configuring Docker nvidia runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# ── 4. Restart Docker ─────────────────────────────────────────────────────────
echo "==> Restarting Docker..."
sudo systemctl restart docker

echo ""
echo "Done! Verify GPU access in Docker with:"
echo "  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "Then start woofalytics with:"
echo "  docker compose up --build"
