#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv command not found" >&2
  exit 1
fi

TORCH_CUDA="$($PYTHON_BIN - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"

if [[ -z "$TORCH_CUDA" ]]; then
  echo "Torch CUDA version unavailable; install a CUDA-enabled torch build first." >&2
  exit 2
fi

NVCC_BIN=""
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  NVCC_BIN="${CUDA_HOME}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC_BIN="$(command -v nvcc)"
fi

if [[ -z "$NVCC_BIN" ]]; then
  cat >&2 <<MSG
nvcc was not found.
flash-attn requires a local CUDA toolkit with nvcc matching torch.version.cuda.
Current torch.version.cuda=$TORCH_CUDA
Install a matching CUDA toolkit and rerun this script.
MSG
  exit 3
fi

NVCC_CUDA="$("$NVCC_BIN" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
if [[ -z "$NVCC_CUDA" ]]; then
  echo "Could not parse nvcc version from: $NVCC_BIN --version" >&2
  exit 4
fi

TORCH_CUDA_MAJOR="${TORCH_CUDA%%.*}"
NVCC_CUDA_MAJOR="${NVCC_CUDA%%.*}"

if [[ "$NVCC_CUDA_MAJOR" != "$TORCH_CUDA_MAJOR" ]]; then
  cat >&2 <<MSG
CUDA mismatch:
  torch.version.cuda = $TORCH_CUDA
  nvcc version      = $NVCC_CUDA
Set CUDA_HOME to a toolkit with the same CUDA major version as torch, then rerun.
MSG
  exit 5
fi

if [[ "$NVCC_CUDA" != "$TORCH_CUDA" ]]; then
  cat >&2 <<MSG
Warning: minor CUDA mismatch detected.
  torch.version.cuda = $TORCH_CUDA
  nvcc version      = $NVCC_CUDA
PyTorch extension builds usually allow minor-version mismatches within the same major CUDA version.
MSG
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/.uv-cache}"
export UV_CACHE_DIR
FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

echo "Building flash-attn against CUDA_HOME=$CUDA_HOME"
CUDA_HOME="$CUDA_HOME" FLASH_ATTENTION_FORCE_BUILD="$FLASH_ATTENTION_FORCE_BUILD" MAX_JOBS="${MAX_JOBS:-1}" \
  uv pip install --python "$PYTHON_BIN" flash-attn --no-build-isolation -v

echo "flash-attn install complete."
