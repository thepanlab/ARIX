#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${CODE_DIR}/data}"
PYTHON="${PYTHON:-python3}"

REFT_REPO=https://github.com/lqtrung1998/mwp_ReFT.git
TRIALGPT_REPO=https://github.com/ncbi-nlp/TrialGPT.git
TRIALGPT_FTP=https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT
QWEN_MATH_DATA=https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/math

TMP_DIR=""
cleanup() { if [ -n "${TMP_DIR}" ]; then rm -rf "${TMP_DIR}"; fi; }
trap cleanup EXIT

log() { printf '[download_data] %s\n' "$*" >&2; }

clone() {
  if [ -d "$2/.git" ]; then
    log "$2 exists, not cloned again"
  else
    git clone --depth 1 "$1" "$2"
  fi
}

fetch() {
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

math_data() {
  clone "${REFT_REPO}" "${DATA_DIR}/mwp_ReFT"
  local name
  for name in gsm8k_python_sdp gsm8k_nl gsm8k_test_set svamp_python_sdp svamp_nl svamp_test_set; do
    ln -sfn "mwp_ReFT/data/${name}.json" "${DATA_DIR}/${name}.json"
  done
  log "ReFT files linked: ${DATA_DIR}/{gsm8k,svamp}_{python_sdp,nl,test_set}.json"

  if [ -d "${DATA_DIR}/MATH/train" ] && [ -d "${DATA_DIR}/MATH/test" ]; then
    log "${DATA_DIR}/MATH exists, not downloaded again"
  else
    TMP_DIR="$(mktemp -d)"
    fetch "${QWEN_MATH_DATA}/train.jsonl" "${TMP_DIR}/train.jsonl"
    fetch "${QWEN_MATH_DATA}/test.jsonl" "${TMP_DIR}/test.jsonl"
    "${PYTHON}" -I - "${DATA_DIR}/MATH" "${TMP_DIR}/train.jsonl" "${TMP_DIR}/test.jsonl" <<'EOF'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
for source, split, expected in zip(sys.argv[2:], ("train", "test"), (7500, 5000)):
    count = 0
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            subject = re.sub(r"[^a-z]+", "_", record["subject"].lower().replace("&", "and")).strip("_")
            if not subject:
                raise SystemExit(f"MATH {split}, line {count + 1}: no subject")
            target = root / split / subject / f"{count}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            item = {"problem": record["problem"], "level": record["level"], "type": record["subject"]}
            item["solution"] = record["solution"]
            target.write_text(json.dumps(item, ensure_ascii=False, indent=4), encoding="utf-8")
            count += 1
    if count != expected:
        raise SystemExit(f"MATH {split}: {count} problems, expected {expected}")
    print(f"MATH {split}: {count} problems written to {root / split}")
EOF
  fi

  if [ ! -s "${DATA_DIR}/aime_2025.jsonl" ]; then
    log "AIME 2025 is not downloaded: write the 30 problems of AIME 2025 I and II to ${DATA_DIR}/aime_2025.jsonl," \
      'one {"problem": ..., "answer": ...} object per line'
  fi
}

clinical_data() {
  clone "${TRIALGPT_REPO}" "${DATA_DIR}/trialgpt"
  local corpus target
  for corpus in trec_2021 trec_2022; do
    target="${DATA_DIR}/trialgpt/dataset/${corpus}/corpus.jsonl"
    if [ -s "${target}" ]; then
      log "${target} exists, not downloaded again"
    elif ! fetch "${TRIALGPT_FTP}/${corpus}_corpus.jsonl" "${target}"; then
      rm -f "${target}.part"
      log "WARNING: could not download ${corpus}_corpus.jsonl from ${TRIALGPT_FTP}; judged pairs whose" \
        "trial is missing from retrieved_trials.json will be skipped (counted in data_summary.json)"
    fi
  done

  if ! "${PYTHON}" -m nltk.downloader -q punkt_tab punkt; then
    log "WARNING: the NLTK punkt model was not installed; patient notes will be split by a regular" \
      "expression whose sentence numbers can differ from those cited by TrialGPT's explanations"
  fi

  mkdir -p "${DATA_DIR}/trialgpt/results"
  for corpus in sigir trec_2021 trec_2022; do
    target="${DATA_DIR}/trialgpt/results/matching_results_${corpus}_gpt-4-turbo.json"
    if [ -s "${target}" ]; then
      log "found ${target}"
    else
      log "missing ${target}: copy the TrialGPT-Matching outputs (GPT-4 Turbo) there, or produce them with" \
        "'python trialgpt_matching/run_matching.py ${corpus} gpt-4-turbo' in ${DATA_DIR}/trialgpt"
    fi
  done
}

main() {
  local parts=("$@")
  [ ${#parts[@]} -gt 0 ] || parts=(math clinical)
  mkdir -p "${DATA_DIR}"
  local part
  for part in "${parts[@]}"; do
    case "${part}" in
      math) math_data ;;
      clinical) clinical_data ;;
      *) log "unknown part '${part}' (expected: math, clinical)"; exit 2 ;;
    esac
  done
}

main "$@"
