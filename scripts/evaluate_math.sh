#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  echo "usage: scripts/evaluate_math.sh MODEL [MODEL ...] [-- EVALUATE_OPTIONS ...]"
  exit "$1"
}

[ $# -ge 1 ] || usage 2
case "$1" in -h | --help) usage 0 ;; esac

models=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  models+=("$1")
  shift
done
[ $# -eq 0 ] || shift
[ ${#models[@]} -ge 1 ] || usage 2

gpus="${NUM_GPUS:-$({ nvidia-smi -L 2>/dev/null || true; } | wc -l)}"
if [ "$((gpus))" -gt 1 ]; then
  launcher=(accelerate launch --multi_gpu --num_processes "$((gpus))" --num_machines 1 --mixed_precision no
    --dynamo_backend no -m math_reasoning.evaluate)
else
  launcher=(python -m math_reasoning.evaluate)
fi

for model in "${models[@]}"; do
  command=("${launcher[@]}" --model "${model}" "$@")
  echo "+ ${command[*]}" >&2
  [ -n "${DRY_RUN:-}" ] || "${command[@]}"
done
