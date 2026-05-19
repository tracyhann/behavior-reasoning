#!/usr/bin/env bash
# Run feature extraction + reasoning for multiple VLM backends across all
# dataset-ours videos, then emit a side-by-side comparison.
#
# Usage:
#   ./run_compare.sh qwen25_72b   # only Qwen2.5-VL-72B
#   ./run_compare.sh qwen3_8b
#   ./run_compare.sh gemma3_12b   # requires HF_TOKEN env (Gemma is gated)
#   ./run_compare.sh all
set -euo pipefail

CONFIG="${1:-all}"
PROJECT_ROOT="/DATA/zihao/projects/Project-Ava/behavior-reasoning"
QWEN_CHECKPOINTS="/DATA/lulin/egotl/Qwen3-VL/checkpoints"
QWEN25_32B_DIR="/DATA/zihao/hf-cache/hub"

# Map of video id -> expected number of subjects (best-effort priors).
declare -A EXPECTED_SUBJECTS=(
  ["oCkUyjaZuNI"]="3"
  ["-JUtzfuiV08"]="2"
  ["DmSmN-oqFZ0"]="3"
  ["G6f0w5BRasw"]="3"
)

USERHOME="/tmp/userhome_bavh"
mkdir -p "$USERHOME"

run_one_video () {
  local backend="$1" model_path_in_container="$2" out_subdir="$3" vid="$4" gpus="$5" extra_env="${6:-}"
  local input="/workspace/dataset-ours/${vid}/${vid}.mp4"
  local out="/workspace/behavior-reasoning/${out_subdir}"
  local expected="${EXPECTED_SUBJECTS[$vid]}"

  echo "=== [$out_subdir] $vid ==="
  docker run --rm --gpus "\"device=${gpus}\"" --ipc=host --shm-size=16g \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${QWEN_CHECKPOINTS}:/models_qwen" \
    -v "${QWEN25_32B_DIR}:/models_qwen25_32b" \
    -v "${USERHOME}:/userhome" \
    -e HOME=/userhome \
    -e YOLO_CONFIG_DIR=/userhome/Ultralytics \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ${extra_env} \
    -w /workspace/behavior-reasoning/feature-extraction \
    --user "$(id -u):$(id -g)" \
    qwen:latest \
    bash -lc "
      mkdir -p /userhome/Ultralytics
      python run_multi.py \
        --backend ${backend} \
        --model-id ${model_path_in_container} \
        --input ${input} \
        --output ${out} \
        --segment-sec 10 --stride-sec 6 \
        --dtype bfloat16 \
        --max-new-tokens 700 \
        --expected-subjects ${expected} \
        --det-model yolov8n.pt --pose-model yolov8n-pose.pt
    "
}

run_reasoning () {
  local out_subdir="$1" vid="$2" gpus="$3" reasoner_model="$4"
  local base="/workspace/behavior-reasoning/${out_subdir}/${vid}"
  echo "=== reasoning [$out_subdir] $vid ==="
  docker run --rm --gpus "\"device=${gpus}\"" --ipc=host --shm-size=16g \
    -v "${PROJECT_ROOT}:/workspace" \
    -v "${QWEN_CHECKPOINTS}:/models_qwen" \
    -v "${USERHOME}:/userhome" \
    -e HOME=/userhome \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -w /workspace/behavior-reasoning/reasoning \
    --user "$(id -u):$(id -g)" \
    qwen:latest \
    bash -lc "
      python scripts/build_behavior_chains.py \
        --characters ${base}/characters/characters.json \
        --detections ${base}/detections/detections.json \
        --out ${base}/chains/behavior_chains.json \
        --canonical-events-out ${base}/chains/canonical_events.json \
        --qa-report-out ${base}/chains/qa_qc_report.json \
        --qa-iterations 5 \
        --reasoner-model ${reasoner_model} \
        --reasoner-dtype bfloat16 \
        --reasoner-max-new-tokens 256
    "
}

VIDS=(oCkUyjaZuNI -JUtzfuiV08 DmSmN-oqFZ0 G6f0w5BRasw)

case "$CONFIG" in
  qwen25_72b|all)
    MODEL=/models_qwen/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5
    OUT=outputs_qwen25_72b
    for v in "${VIDS[@]}"; do
      run_one_video qwen25 "$MODEL" "$OUT" "$v" "4,5,6,7"
    done
    ;;&
  qwen3_8b|all)
    MODEL=/models_qwen/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
    OUT=outputs_qwen3_8b
    for v in "${VIDS[@]}"; do
      run_one_video qwen3 "$MODEL" "$OUT" "$v" "4"
    done
    ;;&
  gemma3_12b|all)
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "Gemma 3 is gated — set HF_TOKEN to download. Skipping."
    else
      MODEL=google/gemma-3-12b-it
      OUT=outputs_gemma3_12b
      for v in "${VIDS[@]}"; do
        run_one_video gemma3 "$MODEL" "$OUT" "$v" "5" "-e HF_TOKEN=${HF_TOKEN}"
      done
    fi
    ;;
esac
