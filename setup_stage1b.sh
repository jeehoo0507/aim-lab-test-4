#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Unconditional paths: inherited global settings must not escape this clone.
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export UV_PYTHON_BIN_DIR="$PWD/.cache/python-bin"
export UV_TOOL_DIR="$PWD/.cache/uv-tools"
export UV_TOOL_BIN_DIR="$PWD/.cache/uv-tool-bin"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_LINK_MODE=copy
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export TMPDIR="$PWD/.cache/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/.cache/huggingface"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p logs "$MPLCONFIGDIR" "$TMPDIR"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 없습니다. 기존 서버에서 사용한 uv 실행 경로를 PATH에 추가해 주세요. sudo는 사용하지 않습니다."
  exit 1
fi
uv python install 3.11 --no-bin
uv sync --frozen --managed-python --python 3.11
mode="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
task_log="logs/setup_stage1b_${mode}_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$task_log") 2>&1
echo "Project: $PWD"
echo "Log: $PWD/$task_log"
case "$mode" in
  install) ;;
  test|plan|verify-flip|run|summary) .venv/bin/python -u -m scripts.plan_stage1b "$mode" "$@" ;;
  *) echo "Usage: bash setup_stage1b.sh [install|test|plan|verify-flip|run|summary]"; exit 2 ;;
esac
