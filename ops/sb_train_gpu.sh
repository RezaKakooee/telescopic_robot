#!/bin/bash
#SBATCH --job-name=radial-gpu
#SBATCH --qos=rtx4090-1day
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


PROJECT_ROOT="/scicore/home/graber0001/kakooe0000/telescopic_robot"

# Usage: sbatch ops/sb_train.sh [train_rl|eval_rl] [config.yaml] [args / key=value ...]
#   arg1: script — a name resolved under scripts/rl/ (default train_rl),
#         or an explicit path like scripts/rl/eval_rl.py
#   arg2: optional config variant — exported as RADIAL_SPHERE_CONFIG so parallel
#         sweep jobs each read their own config instead of the project default
#   rest: passed to the script (argparse flags and hydra-style overrides), e.g.
#         sbatch ops/sb_train.sh train_rl configs/rl/obstacle.yaml rl.total_steps=5e5
#         sbatch ops/sb_train.sh train_rl "" --kind obstacle
PY_SCRIPT="${1:-train_rl}"
case "$PY_SCRIPT" in
    */*) ;;                                  # explicit path: use as given
    *)   PY_SCRIPT="scripts/rl/${PY_SCRIPT}" ;;
esac
PY_SCRIPT="${PY_SCRIPT%.py}"
CFG_ARG="${2:-}"
CFG_TAG=""
if [ -n "$CFG_ARG" ] && [[ "$CFG_ARG" != -* ]] && [[ "$CFG_ARG" != *=* ]]; then
    export RADIAL_SPHERE_CONFIG="$(readlink -f "$CFG_ARG")"
    CFG_TAG="__$(basename "$CFG_ARG" .yaml)"
    EXTRA_ARGS=("${@:3}")
else
    EXTRA_ARGS=()                   # no config given; rest is script args
    for a in "${@:2}"; do [ -n "$a" ] && EXTRA_ARGS+=("$a"); done
fi


run_job() {
    echo "========================================"
    echo "RadialSphere RL Training"
    echo "========================================"
    echo "Job ID : $SLURM_JOB_ID"
    echo "Script : $PY_SCRIPT"
    echo "Config : ${RADIAL_SPHERE_CONFIG:-configs/rl/config.yaml (default)}"
    echo "Node   : $SLURM_NODELIST"
    echo "Start  : $(date)"
    echo ""

    module load CUDA/12.1 2>/dev/null || true
    export MUJOCO_GL=egl
    export PYTHONNOUSERSITE=1

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate roboverse

    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    echo "Python : $(which python3)"
    echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
    echo ""

    python3 -u "$PROJECT_ROOT/$PY_SCRIPT".py "${EXTRA_ARGS[@]}"

    echo ""
    echo "End : $(date)"
    echo "Done."
}



# Mint ONE run id up front (see radial_sphere/run_id.py): the .out log, the
# storage_local run dir, and the wandb run all share this exact name.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output_dir="${PROJECT_ROOT}/storage_local/sci_out"
current_date=$(date +%Y%m%d_%H%M)
job_id=${SLURM_JOB_ID}

export RADIAL_SPHERE_RUN_ID="${current_date}__${job_id}__$(basename "$PY_SCRIPT")${CFG_TAG}"
output_file="${output_dir}/${RADIAL_SPHERE_RUN_ID}.out"

mkdir -p ${output_dir}

run_job > "${output_file}" 2>&1
