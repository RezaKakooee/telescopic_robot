#!/usr/bin/env bash
# Watch training runs, wherever they happen to be running.
#
# The old version of this file was one line, `watch -n 5 squeue -u $USER`, which
# only ever worked on a Slurm cluster. On a plain box there is no `squeue`, so
# it printed an error and showed nothing at all -- including when a run really
# was going.
#
# So: use Slurm when Slurm is there, and otherwise show the local processes and
# what their logs say. Runs started by `ops/local_train.sh`, by `nohup`, or from
# an editor terminal all look the same to this.
#
#   ops/watch.sh            follow, refreshing every 5 s
#   ops/watch.sh once       print one snapshot and exit
#   ops/watch.sh 15         follow, refreshing every 15 s
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL=5
ONCE=0
case "${1:-}" in
    once) ONCE=1 ;;
    ''  ) ;;
    *[!0-9]*) echo "usage: ops/watch.sh [once|SECONDS]" >&2; exit 2 ;;
    *) INTERVAL="$1" ;;
esac

#: Anything that looks like one of our training or evaluation entry points.
PATTERN='scripts/(rl|imitation|data)/[a-z_]*\.py|train_maze_bank|train_rl|train_mujoco_rl|train_finetune_rl|eval_rl|eval_maze_bank'

snapshot() {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S')   $(hostname) ==="

    if command -v squeue >/dev/null 2>&1; then
        echo
        echo "-- slurm queue --"
        squeue -u "$USER"
        return
    fi

    echo
    echo "-- local training processes --"
    # `ps` matches its own grep, so drop that and this script's own line.
    local rows
    rows="$(ps -eo pid,etime,pcpu,pmem,cmd --sort=start_time \
            | grep -E "$PATTERN" | grep -v 'grep -E' | grep -v 'watch.sh')"
    if [ -z "$rows" ]; then
        echo "   nothing running"
    else
        printf "%8s %11s %7s %6s  %s\n" PID ELAPSED CPU% MEM% SCRIPT
        echo "$rows" | while read -r pid etime pcpu pmem cmd; do
            # The interpreter path and the scripts/rl/ prefix are the same on
            # every line and push the useful part off the screen. Keep the
            # script's own name and its key=value arguments, drop the rest.
            local short
            short="$(printf '%s' "$cmd" \
                     | sed -E 's#^\S*python[0-9.]*( +-[A-Za-z]+)* +##; s#^\S*/##')"
            printf "%8s %11s %6s%% %5s%%  %s\n" \
                   "$pid" "$etime" "$pcpu" "$pmem" "${short:0:70}"
        done
        echo
        echo "   CPU% above 100 is normal: one worker per vectorised env."
    fi

    echo
    echo "-- newest run directories --"
    # Progress lives in the run dir, so show what is there rather than guessing
    # where the caller sent stdout.
    local dirs
    dirs="$(ls -1dt "$PROJECT_ROOT"/storage_local/*train* 2>/dev/null | head -3)"
    if [ -z "$dirs" ]; then
        echo "   none under storage_local/"
    else
        for d in $dirs; do
            local ck last
            ck="$(ls -1 "$d"/checkpoints/*.zip 2>/dev/null | wc -l | tr -d ' ')"
            last="$(ls -1t "$d"/checkpoints/*.zip 2>/dev/null | head -1)"
            printf "   %-58s %s ckpt" "$(basename "$d")" "$ck"
            [ -n "$last" ] && printf "   latest %s (%s)" \
                "$(basename "$last")" "$(date -r "$last" '+%H:%M:%S')"
            printf "\n"
            # `train.log` is written by the hydra-based trainers; the newer ones
            # log to stdout, so this is best effort rather than guaranteed.
            if [ -s "$d/train.log" ]; then
                sed -n '$p' "$d/train.log" | sed 's/^/      /'
            fi
        done
    fi

    echo
    echo "-- tensorboard --"
    echo "   tensorboard --logdir $PROJECT_ROOT/storage_local --port 6006"
}

if [ "$ONCE" -eq 1 ]; then
    snapshot
    exit 0
fi

if command -v watch >/dev/null 2>&1; then
    export -f snapshot 2>/dev/null || true
    while true; do
        clear
        snapshot
        sleep "$INTERVAL"
    done
else
    snapshot
fi
