#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Usage: ops/local_train.sh [train_rl|eval_rl] [config.yaml] [args / key=value ...]
PY_SCRIPT="${1:-train_rl}"
case "$PY_SCRIPT" in
    */*) ;;
    *) PY_SCRIPT="scripts/rl/$PY_SCRIPT" ;;
esac
PY_SCRIPT="${PY_SCRIPT%.py}"

CFG_ARG="${2:-}"
if [[ -n "$CFG_ARG" && "$CFG_ARG" != -* && "$CFG_ARG" != *=* ]]; then
    export RADIAL_SPHERE_CONFIG="$(readlink -f "$CFG_ARG")"
    EXTRA_ARGS=("${@:3}")
else
    EXTRA_ARGS=()
    for arg in "${@:2}"; do [ -n "$arg" ] && EXTRA_ARGS+=("$arg"); done
fi

run_job() {
    echo "========================================"
    echo "RadialSphere RL Training"
    echo "========================================"
    echo "Script : $PY_SCRIPT"
    echo "Config : ${RADIAL_SPHERE_CONFIG:-configs/rl/config.yaml (default)}"
    echo "Start  : $(date)"
    echo ""

    export MUJOCO_GL=egl
    export PYTHONNOUSERSITE=1

    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate roboverse

    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

    echo "Python : $(which python3)"
    echo ""

    cd "$PROJECT_ROOT"
    python3 -u "$PY_SCRIPT.py" "${EXTRA_ARGS[@]}"
}

if [ "${RADIAL_LOCAL_WORKER:-0}" != 1 ]; then
    output_dir="$PROJECT_ROOT/storage_local/sci_out"
    run_id="$(date +%Y%m%d_%H%M)__local_$$__$(basename "$PY_SCRIPT")"
    output_file="$output_dir/$run_id.out"

    mkdir -p "$output_dir"
    export RADIAL_LOCAL_WORKER=1
    export RADIAL_SPHERE_RUN_ID="$run_id"
    nohup setsid "$0" "$@" > "$output_file" 2>&1 < /dev/null &
    pid=$!

    echo "Started : $run_id"
    echo "PID     : $pid"
    echo "Log     : $output_file"
    echo "Follow  : tail -f '$output_file'"
    echo "Stop    : kill -- -$pid"
    exit 0
fi

run_job
