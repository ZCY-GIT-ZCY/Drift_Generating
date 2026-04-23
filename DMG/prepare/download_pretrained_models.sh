#!/usr/bin/env bash
set -euo pipefail

# DMG: Download Pretrained Models (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Preparing directories under: ${DMG_ROOT}"
mkdir -p "${DMG_ROOT}/deps"
mkdir -p "${DMG_ROOT}/pretrained_models"

echo
echo "Please download and place the following assets:"
echo "1) MLD HumanML3D checkpoint (contains VAE weights)"
echo "   URL : https://drive.google.com/file/d/1hplrnQwUK_cZFHirZIOuVP0RSyZEC1YM/view"
echo "   Dest: ${DMG_ROOT}/pretrained_models/mld_vae_humanml3d.ckpt"
echo
echo "2) CLIP model cache directory"
echo "   Dest: ${DMG_ROOT}/deps/clip-vit-large-patch14"
echo "   Note: also auto-downloaded on first run if internet is available."
echo
echo "3) T2M evaluators"
echo "   URL : https://drive.google.com/file/d/1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8/view"
echo "   Direct URL : https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
echo "   Dest: ${DMG_ROOT}/deps/t2m"
echo
echo "Alternative source: https://github.com/ChenFengYe/motion-latent-diffusion"
echo
echo "Done."
