#!/usr/bin/env bash
set -euo pipefail

# DMG: Prepare T2M evaluator directory (Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${DMG_ROOT}/deps/t2m"

echo "Prepared: ${DMG_ROOT}/deps/t2m"
echo "Please place evaluator files in this directory."
echo "Download URL:"
echo "- https://drive.google.com/file/d/1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8/view"
echo "Direct URL (for gdown/wget tools that support it):"
echo "- https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
