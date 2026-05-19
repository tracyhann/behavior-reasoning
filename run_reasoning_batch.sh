#!/usr/bin/env bash
# Run reasoning + QA/QC for every video under a given feature-extraction output dir.
# Reuses one docker container per output dir (model loaded once per video,
# but at 7B it's quick).
set -euo pipefail

OUT_DIR="${1:-}"
if [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 <output-subdir under behavior-reasoning/>"
  exit 1
fi

PROJECT_ROOT="/DATA/zihao/projects/Project-Ava/behavior-reasoning"
QWEN_CHECKPOINTS="/DATA/lulin/egotl/Qwen3-VL/checkpoints"
USERHOME="/tmp/userhome_bavh"
REASONER_MODEL="/models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
GPUS="${GPUS:-6}"

VIDS=(oCkUyjaZuNI -JUtzfuiV08 DmSmN-oqFZ0 G6f0w5BRasw)

cmd=""
for v in "${VIDS[@]}"; do
  base="/workspace/behavior-reasoning/${OUT_DIR}/${v}"
  cmd+="
  echo '=== reasoning ${v} ===';
  if [ -f \"${base}/characters/characters.json\" ]; then
    python scripts/build_behavior_chains.py \
      --characters ${base}/characters/characters.json \
      --detections ${base}/detections/detections.json \
      --out ${base}/chains/behavior_chains.json \
      --canonical-events-out ${base}/chains/canonical_events.json \
      --qa-report-out ${base}/chains/qa_qc_report.json \
      --qa-iterations 5 \
      --reasoner-model ${REASONER_MODEL} \
      --reasoner-dtype bfloat16 \
      --reasoner-max-new-tokens 256;
  else
    echo '  no characters.json, skipping';
  fi
  "
done

docker run --rm --gpus "\"device=${GPUS}\"" --ipc=host --shm-size=16g \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${QWEN_CHECKPOINTS}:/models" \
  -v "${USERHOME}:/userhome" \
  -e HOME=/userhome \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -w /workspace/behavior-reasoning/reasoning \
  --user "$(id -u):$(id -g)" \
  qwen:latest \
  bash -lc "$cmd"
