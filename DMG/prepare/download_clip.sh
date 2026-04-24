#!/usr/bin/env bash
set -euo pipefail

# DMG: Prepare CLIP model directory (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${DMG_ROOT}/deps/clip"

echo "Prepared: ${DMG_ROOT}/deps/clip"
echo "Please place ViT-B-32 weights at: ${DMG_ROOT}/deps/clip/ViT-B-32.pt"
echo "Download URL: https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin"
echo "(rename pytorch_model.bin to ViT-B-32.pt)"
