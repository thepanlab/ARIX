#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  echo "usage: scripts/run_ablations_math.sh GROUP [GROUP ...] [key.sub=value ...] (GROUP: baselines bank rules embedding group seeds components negative all)"
  exit "$1"
}

BENCHMARKS=(svamp gsm8k math)
MODELS=(qwen25_7b gptoss_20b)
SEEDS=(2222 3333 4444 5555)
GROUPS_IN_ORDER=(baselines bank rules embedding group seeds components negative)

train() { scripts/train_math.sh "$@" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}; }

run_group() {
  local benchmark model overlay size seed
  case "$1" in
    baselines)
      for model in "${MODELS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_${model}" grpo; done
      done
      for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_qwen25_7b" dr_grpo; done
      ;;
    bank)
      for model in "${MODELS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_${model}" arix_wo_rb; done
      done
      ;;
    rules)
      for overlay in random_retrieval fifo_update random_update; do
        for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_qwen25_7b" "${overlay}"; done
      done
      ;;
    embedding)
      for overlay in mean_pooling proj32; do
        for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_qwen25_7b" "${overlay}"; done
      done
      ;;
    group)
      for benchmark in "${BENCHMARKS[@]}"; do
        train "${benchmark}_qwen25_7b" g8
        train "${benchmark}_qwen25_7b" k10
        for size in 4 8 16 32; do
          train "${benchmark}_qwen25_7b" arix_wo_rb "training.group_size=${size}" \
            "output.output_dir=outputs/math/${benchmark}_qwen25_7b-arix_wo_rb-g${size}"
        done
      done
      ;;
    seeds)
      for benchmark in "${BENCHMARKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          train "${benchmark}_qwen25_7b" "training.seed=${seed}" \
            "output.output_dir=outputs/math/${benchmark}_qwen25_7b-seed${seed}"
          train "${benchmark}_qwen25_7b" arix_wo_rb "training.seed=${seed}" \
            "output.output_dir=outputs/math/${benchmark}_qwen25_7b-arix_wo_rb-seed${seed}"
        done
      done
      ;;
    components)
      for model in "${MODELS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_${model}" no_partial_reward; done
      done
      for benchmark in "${BENCHMARKS[@]}"; do train "${benchmark}_qwen25_7b" no_contrastive; done
      ;;
    negative)
      train math_qwen25_7b negative_loss
      ;;
    *)
      echo "unknown group: $1" >&2
      usage 2
      ;;
  esac
}

[ $# -ge 1 ] || usage 2
case "$1" in -h | --help) usage 0 ;; esac
groups=()
EXTRA_OVERRIDES=()
for arg in "$@"; do
  case "${arg}" in
    *=*) EXTRA_OVERRIDES+=("${arg}") ;;
    *) groups+=("${arg}") ;;
  esac
done
[ ${#groups[@]} -ge 1 ] || usage 2
[ "${groups[*]}" != "all" ] || groups=("${GROUPS_IN_ORDER[@]}")
for group in "${groups[@]}"; do
  run_group "${group}"
done
