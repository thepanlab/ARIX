#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  echo "usage: scripts/train_clinical.sh RUN [OVERLAY ...] [key.sub=value ...]"
  exit "$1"
}

config_path() {
  case "$1" in
    *.yaml | *.yml) printf '%s\n' "$1" ;;
    *) printf '%s/%s.yaml\n' "$2" "$1" ;;
  esac
}

[ $# -ge 1 ] || usage 2
case "$1" in -h | --help) usage 0 ;; esac

base="$(config_path "$1" clinical_trials/configs)"
shift
configs=("${base}")
overrides=()
for arg in "$@"; do
  case "${arg}" in
    *=*) overrides+=("${arg}") ;;
    *) configs+=("$(config_path "${arg}" clinical_trials/configs/ablations)") ;;
  esac
done
config_args=()
for config in "${configs[@]}"; do
  [ -f "${config}" ] || { echo "config file not found: ${config}" >&2; exit 2; }
  config_args+=(--config "${config}")
done

case "$(basename "${base}")" in
  *deepseek_r1_7b*) default_accelerate=configs/accelerate/zero2_6gpu.yaml ;;
  *) default_accelerate=configs/accelerate/zero3_6gpu.yaml ;;
esac
command=(accelerate launch --config_file "${ACCELERATE_CONFIG:-${default_accelerate}}"
  -m clinical_trials.train "${config_args[@]}" ${overrides[@]+"${overrides[@]}"})

echo "+ ${command[*]}" >&2
[ -n "${DRY_RUN:-}" ] || "${command[@]}"
