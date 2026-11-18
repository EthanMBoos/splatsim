#!/usr/bin/env bash
# Run splatsim on Georgia Tech PACE ICE. See docs/PACE.md.
#
#   scripts/pace.sh setup                  install uv and splatsim's environment
#   scripts/pace.sh trainer                install 3DGRUT, in a GPU job
#   scripts/pace.sh cpu CMD...             queue CMD on a CPU node
#   scripts/pace.sh train NAME DATA [ARGS] queue a training; result in runs/NAME
#   scripts/pace.sh replay CKPT DATA OUT [--shift X Y Z]
#                                          queue a replay of DATA in a trained splat
#   scripts/pace.sh run CMD...             run CMD here, with splatsim's environment
#   scripts/pace.sh node | gpu [CARD]      a shell on a CPU or GPU node
#   scripts/pace.sh cards                  the card types and what is free right now
#   scripts/pace.sh hf ARGS...             the Hugging Face CLI, run in data/
#   scripts/pace.sh space                  quota, and the size of data and runs
#   scripts/pace.sh clean runs|data|build  delete one of them, after asking
#
# CARD is h100 (default), l40s or a40. Set it for train and replay with
# SPLATSIM_CARD. Every job's log is ~/scratch/splatsim/logs/JOBID.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Everything big lives here: both environments, and the user's data/ and runs/.
ROOT="${SPLATSIM_SCRATCH:-$HOME/scratch/splatsim}"
TRAINER="$ROOT/3dgrut"
# The commit the numbers in docs/TRACKS.md were measured with.
TRAINER_COMMIT=dfd3d54c8c612003fb7948443edf7778f2bcd5ec
# The CUDA 12.6 compiler on ICE rejects compute_100 (Blackwell), which
# 3DGRUT's installer asks for by default. This list covers A40, L40S and H100,
# so a Blackwell cannot train. Training needs no RT cores, and H100s are the
# cards ICE has most of.
ARCH_LIST="8.0;8.6;8.9;9.0"
CUDA_MODULE=cuda/12.6.1

CARDS="h100 l40s a40"

# ICE GPU types: card name -> partition and Slurm feature, from sinfo on
# 2026-11-14. The partition must be named. Otherwise ICE adds partitions with
# no matching node and the job stays at BadConstraints.
card_request() {
    EXCLUDE=
    case "$1" in
    h100)
        PARTITION=coe-gpu FEATURE=gpu-h100
        # One card on each of these nodes failed at startup on every run on
        # 2026-11-14, and Slurm returns the same card. PACE may repair them,
        # so retry without this line occasionally.
        EXCLUDE=atl1-1-03-010-10-0,atl1-1-03-010-25-0
        ;;
    l40s) PARTITION=ice-gpu FEATURE=gpu-l40s ;;
    a40) PARTITION=ice-gpu FEATURE=gpu-a40 ;;
    *) echo "Unknown card '$1'. Choose from: $CARDS" >&2; exit 1 ;;
    esac
}

make_dirs() {
    mkdir -p "$ROOT"/{uv/bin,uv/cache,uv/python,logs,data,runs}
    ROOT="$(cd "$ROOT" && pwd -P)"
    TRAINER="$ROOT/3dgrut"
    export PATH="$ROOT/uv/bin:$PATH"
    export UV_CACHE_DIR="$ROOT/uv/cache"
    export UV_PYTHON_INSTALL_DIR="$ROOT/uv/python"
    # Home is 30 GB, so the environment is in scratch, not in the repo.
    export UV_PROJECT_ENVIRONMENT="$ROOT/env"
}

# Run a command with 3DGRUT's environment active, from its folder. Its
# activate script and `module` both read unset variables.
in_trainer() {
    [ -d "$TRAINER/.venv" ] || { echo "No 3DGRUT at $TRAINER. Run: scripts/pace.sh trainer" >&2; exit 1; }
    set +u
    module load "$CUDA_MODULE"
    cd "$TRAINER"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    set -u
    "$@"
}

# Queue a command. $1 is "cpu" or a card; the rest is the command.
queue() {
    local where="$1" request
    shift
    if [ "$where" = cpu ]; then
        request=(--cpus-per-task=8 --mem=32G)
    else
        card_request "$where"
        request=(--partition="$PARTITION" --constraint="$FEATURE" --gres=gpu:1
            ${EXCLUDE:+--exclude="$EXCLUDE"} --cpus-per-task=12 --mem=64G)
    fi
    local job
    job="$(sbatch --parsable --job-name="splatsim-$where" "${request[@]}" --time=04:00:00 \
        --chdir="$REPO" --output="$ROOT/logs/%j.log" \
        --wrap="$(printf '%q ' "$@")")"
    job="${job%%;*}" # --parsable can print "id;cluster"
    echo "Submitted job $job on $where. Check the queue with: squeue --me"
    echo "tail -n +1 -f $ROOT/logs/$job.log"
}

# The folder a path names, made absolute: jobs start in other folders.
absolute() {
    mkdir -p "$(dirname "$1")"
    echo "$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")"
}

make_dirs
case "${1:-}" in
setup)
    if [ ! -x "$ROOT/uv/bin/uv" ]; then
        curl -LsSf https://astral.sh/uv/install.sh |
            env UV_INSTALL_DIR="$ROOT/uv/bin" UV_NO_MODIFY_PATH=1 sh
    fi
    (cd "$REPO" && uv sync --extra sfm)
    echo "Setup done. Everything is under $ROOT. Next: scripts/pace.sh trainer"
    ;;
trainer)
    # One of 3DGRUT's packages links against CUDA while it builds, so the
    # install fails on a login or CPU node with "cannot find -lcuda".
    queue "${2:-h100}" scripts/pace.sh install-trainer
    ;;
install-trainer)
    [ -d "$TRAINER/.git" ] || git clone --recursive https://github.com/nv-tlabs/3dgrut "$TRAINER"
    cd "$TRAINER"
    git checkout --quiet "$TRAINER_COMMIT"
    git submodule update --init --recursive
    set +u
    module load "$CUDA_MODULE"
    set -u
    TORCH_CUDA_ARCH_LIST="$ARCH_LIST" ./install_env_uv.sh
    echo "splatsim: trainer installed at $TRAINER"
    ;;
cpu)
    shift
    [ $# -gt 0 ] || { echo "Nothing to run. Usage: scripts/pace.sh cpu CMD..." >&2; exit 1; }
    queue cpu scripts/pace.sh run "$@"
    ;;
train)
    [ $# -ge 3 ] || { echo "Usage: scripts/pace.sh train NAME DATA [ARGS]" >&2; exit 1; }
    name="$2" data="$(absolute "$3")"
    shift 3
    queue "${SPLATSIM_CARD:-h100}" scripts/pace.sh in-trainer python train.py \
        --config-name apps/colmap_3dgut.yaml "path=$data" "out_dir=$ROOT/runs" \
        "experiment_name=$name" export_ply.enabled=true num_workers=8 "$@"
    ;;
replay)
    [ $# -ge 4 ] || { echo "Usage: scripts/pace.sh replay CKPT DATA OUT [--shift X Y Z]" >&2; exit 1; }
    checkpoint="$(absolute "$2")" data="$(absolute "$3")" out="$(absolute "$4")"
    shift 4
    queue "${SPLATSIM_CARD:-h100}" scripts/pace.sh in-trainer python \
        "$REPO/src/splatsim/replay.py" "$checkpoint" "$data" "$out" "$@"
    ;;
in-trainer)
    shift
    in_trainer "$@"
    ;;
run)
    shift
    cd "$REPO"
    exec "$@"
    ;;
node)
    exec srun --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
    ;;
gpu)
    card_request "${2:-h100}"
    exec srun --partition="$PARTITION" --constraint="$FEATURE" --gres=gpu:1 \
        ${EXCLUDE:+--exclude="$EXCLUDE"} --cpus-per-task=12 --mem=64G --time=04:00:00 --pty bash
    ;;
cards)
    # Node counts by state, from Slurm. An idle or mixed node may have a free
    # card; an allocated node has none.
    for card in $CARDS; do
        card_request "$card"
        states="$(sinfo -h -N -p "$PARTITION" -o "%N %T %f" | grep -w "$FEATURE" |
            awk '{print $2}' | sort | uniq -c | awk '{printf "%s %s  ", $1, $2}')"
        printf '%-10s %-11s %-21s %s\n' "$card" "$PARTITION" "$FEATURE" "$states"
    done
    ;;
hf)
    # Runs on the login node: it needs the network. The caches go to scratch
    # because home is 30 GB.
    shift
    cd "$ROOT/data"
    exec uvx --from huggingface_hub hf "$@"
    ;;
space)
    pace-quota 2>/dev/null | grep -E "Usage|Home|Scratch" || true
    du -sh "$ROOT/data" "$ROOT/runs"
    ;;
clean)
    case "${2:-}" in
    runs | data) targets=("$ROOT/$2") ;;
    # Both environments. data/, runs/, logs/ and uv/ are left alone.
    build) targets=("$ROOT"/{env,3dgrut}) ;;
    *) echo "Usage: scripts/pace.sh clean runs|data|build" >&2; exit 1 ;;
    esac
    printf '%s\n' "${targets[@]}"
    read -r -p "Delete these? This cannot be undone. [y/N] " answer
    [ "$answer" = y ] || { echo "Nothing deleted."; exit 0; }
    rm -rf "${targets[@]}"
    if [ "$2" = build ]; then echo "Deleted. Rebuild with: scripts/pace.sh setup"; else mkdir -p "$ROOT/$2"; fi
    ;;
*)
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
