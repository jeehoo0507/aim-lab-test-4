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
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
task_log="logs/setup_${mode}_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$task_log") 2>&1
echo "Project: $PWD"
echo "Log: $PWD/$task_log"
case "$mode" in
  install) ;;
  stage1-test) .venv/bin/python -m scripts.plan_stage1 test "$@" ;;
  stage1-plan) .venv/bin/python -m scripts.plan_stage1 plan "$@" ;;
  stage1-run) .venv/bin/python -m scripts.plan_stage1 run "$@" ;;
  stage1-summary) .venv/bin/python -m scripts.plan_stage1 summary "$@" ;;
  check)
    .venv/bin/python -m pytest -q
    .venv/bin/python run.py smoke --device cpu
    .venv/bin/python run.py check "$@"
    ;;
  prepare)
    .venv/bin/python run.py prepare "$@"
    .venv/bin/python run.py check "$@"
    ;;
  benchmark) .venv/bin/python run.py benchmark "$@" ;;
  pilot)
    .venv/bin/python run.py pipeline --seeds 0 --methods ce full "$@"
    ;;
  run) .venv/bin/python run.py pipeline "$@" ;;
  evaluate) .venv/bin/python run.py evaluate "$@" ;;
  export) .venv/bin/python run.py export "$@" ;;
  anneal) .venv/bin/python -m coco_kd.anneal run "$@" ;;
  anneal-export) .venv/bin/python -m coco_kd.anneal export "$@" ;;
  adaptive) .venv/bin/python scripts/run_adaptive200.py run "$@" ;;
  adaptive-evaluate) .venv/bin/python scripts/run_adaptive200.py evaluate "$@" ;;
  *) echo "Usage: bash setup.sh [install|stage1-test|stage1-plan|stage1-run|stage1-summary|check|prepare|benchmark|pilot|run|evaluate|export|anneal|anneal-export|adaptive|adaptive-evaluate]"; exit 2 ;;
esac
