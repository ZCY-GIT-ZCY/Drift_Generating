#!/usr/bin/env bash
set -euo pipefail

# DMG: Prepare CLIP model directory (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${DMG_ROOT}/deps/clip-vit-large-patch14"

echo "Prepared: ${DMG_ROOT}/deps/clip-vit-large-patch14"
echo "CLIP model files are usually downloaded automatically on first run."
