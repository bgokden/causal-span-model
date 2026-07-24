#!/usr/bin/env bash
#
# End-to-end pipeline: CNC CSV -> BIO -> trained mDeBERTa model -> ONNX.
# Run on your own infra. Each step is a console entry point installed by
# `pip install -e .` (see pyproject [project.scripts]).
#
# The default path trains on CLEAN English CNC and relies on mDeBERTa-v3's
# zero-shot cross-lingual transfer for multilingual coverage. Marker-injection MT
# augmentation (causal-translate) is NOT in the default path: it was empirically
# unreliable through real translation (es ~32% / zh 0% span survival), so it
# would train on corrupt labels. See the README "Multilingual" section; a
# word-alignment projection is the planned robust alternative.
#
# Prerequisites:
#   1. pip install -e '.[onnx,dev]'
#   2. Fetch the Causal News Corpus subtask-2 CSVs (CC0-1.0) from
#      https://github.com/tanfiona/CausalNewsCorpus and place the train/dev
#      CSVs under data/ (paths below).
#
# Override any path with an environment variable, e.g.:
#   OUTPUT_DIR=outputs/run2 EPOCHS=8 ./scripts/run_pipeline.sh

set -euo pipefail

CNC_TRAIN_CSV="${CNC_TRAIN_CSV:-data/cnc_train_subtask2.csv}"
CNC_DEV_CSV="${CNC_DEV_CSV:-data/cnc_dev_subtask2.csv}"

TRAIN_BIO="${TRAIN_BIO:-data/train.bio.jsonl}"
DEV_BIO="${DEV_BIO:-data/dev.bio.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/causal-span-mdeberta}"
BASE_MODEL="${BASE_MODEL:-microsoft/mdeberta-v3-base}"
EPOCHS="${EPOCHS:-5}"
RELATIONS="${RELATIONS:-all}"

echo "== 1/3 Convert CNC CSV to BIO (English) =="
cnc-to-bio "${CNC_TRAIN_CSV}" "${TRAIN_BIO}" --relations "${RELATIONS}"
cnc-to-bio "${CNC_DEV_CSV}" "${DEV_BIO}" --relations "${RELATIONS}"

echo "== 2/3 Train mDeBERTa-v3 token classifier =="
causal-train \
  --train "${TRAIN_BIO}" \
  --eval "${DEV_BIO}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model "${BASE_MODEL}" \
  --epochs "${EPOCHS}"

echo "== 3/3 Export to ONNX (and verify the reasongraph contract) =="
causal-export-onnx "${OUTPUT_DIR}" --check

echo "== Evaluate (per-language + aggregate) =="
causal-evaluate "${OUTPUT_DIR}" "${DEV_BIO}"

echo "Done. Point reasongraph's OnnxTokenClassifierExtractor at ${OUTPUT_DIR}"
