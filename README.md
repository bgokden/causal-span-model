# causal-span-model

A standalone training pipeline for a multilingual **causal span tagger**: given a
sentence, it labels the cause, effect and signal (connective) spans with BIO
tags. It is a separate project from `reasongraph` on purpose -- `reasongraph`
ships the lightweight inference consumer, this repo produces the model.

## Why a model instead of the cue lexicon

`reasongraph` currently splits explicit causal sentences with a direction-aware
cue lexicon (`causal_from_cues`). That is fast and precise on clean, explicit
phrasing, but on free-form prose it over-grabs subordinate clauses and cannot
handle implicit causality. A trained span model reads the cause/effect boundaries
and their direction directly from the text, including reversed phrasing ("the
crash resulted from brake failure"), which the lexicon can only approximate.

The lexicon stays as the fast path; this model is the accurate path that replaces
`gliner-relex-multi` inside `HybridCausalExtractor` once trained.

## Label scheme

Seven BIO labels, with direction encoded in the span TYPE:

```
O
B-CAUSE   I-CAUSE     the cause span   (CNC ARG0)
B-EFFECT  I-EFFECT    the effect span  (CNC ARG1)
B-SIGNAL  I-SIGNAL    the connective   (CNC SIG0/SIG1)
```

The label order is fixed in `causal_span_model/labels.py` and baked into every
exported `config.json`. Encoding direction in the types is what lets the consumer
recover `cause -> effect` without a separate orientation lexicon.

## Data

- **Primary: Causal News Corpus (CNC), subtask 2** -- ~1,900 causal sentences
  with cause/effect/signal spans. License **CC0-1.0**.
  https://github.com/tanfiona/CausalNewsCorpus
  ARG0 = cause, ARG1 = effect, SIG0/SIG1 = signal.
- Optional extensions (not wired by default): BECauSE 2.0 (MIT), FinCausal
  (CC-BY). Add them by converting to the same BIO JSONL shape.

CNC is English-only. Multilingual coverage comes from **mDeBERTa-v3's zero-shot
cross-lingual transfer**: trained on clean English spans, the multilingual encoder
generalizes to other languages at inference. See "Multilingual" below for why we
do NOT machine-translate the training data.

## Base model

`microsoft/mdeberta-v3-base` (MIT, ~100 languages) -- the multilingual encoder
family already used elsewhere in the reasongraph stack (gliner-relex-multi). Token
classification head, standard subword-label alignment (first subword carries the
label). It is contract-compatible with the consumer: `DebertaV2TokenizerFast`
provides `word_ids`/`offset_mapping`, and its ONNX export needs only `input_ids` +
`attention_mask` (no `token_type_ids`). Export uses opset 18 (DeBERTa-v2 minimum).
Override with `--base-model` (e.g. `FacebookAI/xlm-roberta-base`) if desired.

## Pipeline

Install and run each stage (console scripts are declared in `pyproject.toml`):

```bash
pip install -e '.[onnx,dev]'

# 1. CNC CSV -> BIO JSONL  ({"tokens": [...], "tags": [...], "lang": "en"})
cnc-to-bio data/cnc_train_subtask2.csv data/train.bio.jsonl
cnc-to-bio data/cnc_dev_subtask2.csv   data/dev.bio.jsonl

# 2. Fine-tune mDeBERTa-v3-base on clean English (bf16 auto on GPU)
causal-train --train data/train.bio.jsonl --eval data/dev.bio.jsonl \
             --output-dir outputs/causal-span-mdeberta --epochs 5

# 3. Export to ONNX in the layout reasongraph consumes, and verify the contract
causal-export-onnx outputs/causal-span-mdeberta --check

# Evaluate: per-language + aggregate seqeval, plus decoded cause->effect relations
causal-evaluate outputs/causal-span-mdeberta data/dev.bio.jsonl
```

`scripts/run_pipeline.sh` chains these steps with env-var-overridable paths.

### Multilingual: why we do NOT translate the training data

An obvious idea is to machine-translate the English training data and project the
spans onto the translations. `translate.py` implements that (wrap each span in a
sentinel, translate with NLLB-200, parse the sentinels back, discard anything that
doesn't round-trip). **It is kept only as an experimental tool and is NOT in the
default pipeline**, because it does not survive real translation: measured span
survival was ~32% for Spanish and **0% for Chinese** (NLLB rewrites/reorders and
drops the sentinels, especially for non-Latin scripts). Training on that would
mean training on corrupt or absent labels for most languages -- worse than not
augmenting.

Instead, multilingual capability comes from mDeBERTa-v3's zero-shot cross-lingual
transfer off clean English spans. The planned robust alternative to sentinels is
**word-alignment projection** (e.g. SimAlign) for the space-delimited languages,
to be added only if a language's measured quality is insufficient.

## Pointer model (beats the CNC baseline)

The BIO tagger above is reasongraph-friendly but caps around 0.55 F1 on the CNC
Subtask-2 official scorer. `src/causal_span_model/pointer/` is a stronger
**span-pointer** model (start/end pointers per role + a signal classifier +
beam-search top-2 decoding), reimplemented from the CNC baseline on mDeBERTa-v3.

Official scorer (FairEval + best-combination), V2 dev:

| Model | F1 |
|---|---|
| BIO tagger (best-span decode) | 0.55 |
| **Pointer model** | **0.70** |
| Organizer baseline (2023) | 0.627 |
| 2022 winner (test) | 0.542 |
| 2023 winner (test) | 0.728 |

It beats the 0.627 baseline (Cause 0.74 / Effect 0.69 / Signal 0.65; multi-relation
F1 0.52 via beam top-2). Augmented data was tried and hurt (0.675), so it is unused.

### vs a few-shot LLM

Would a prompted general LLM beat the fine-tune? No. Qwen2.5-7B-Instruct, few-shot on
the same dev set and official scorer:

| Approach | Official F1 |
|---|---|
| LLM few-shot, plain causal prompt | 0.24 |
| LLM few-shot, scheme-aware prompt | 0.41 |
| **Pointer model** | **0.70** |

The plain LLM under-detects CNC's broad causality (purpose/motive/implicit are all
labelled causal); a scheme-aware prompt lifts detection but partly by over-flagging
non-causal sentences, which the scorer ignores. On a capability-fair subset (causality
a reader recognises without CNC's scheme) the gap narrows to ~0.45 (LLM) vs ~0.63
(pointer) -- the residual is exact span-boundary precision, which is what fine-tuning
on the annotation buys.

```bash
causal-pointer-train --train data/cnc_train_subtask2.csv --dev data/cnc_dev_subtask2.csv \
                     --output-dir outputs/pointer-mdeberta --epochs 10 --lr 3e-5
causal-pointer-submit outputs/pointer-mdeberta cnc_eval/dev_grouped.csv submission.json
# score with the official scorer under cnc_eval/ (_evaluate.py)
```

This model is a custom architecture: it does NOT fit the generic ONNX consumer
(needs its own beam decoder). It is trained on English spans but **multilingual at
inference** via `pointer/infer.py`'s script-aware segmentation (whitespace tokens
for Latin/Cyrillic/Arabic, characters for CJK/Thai) -- verified on
es/fr/de/pt/tr/ru/ar and zh/ja. It has a built-in **causal gate** (~0.85 accuracy),
so `predict_relations` returns `[]` on non-causal text -- safe on arbitrary input.
reasongraph consumes it via `CausalPointerExtractor`.

## reasongraph integration contract

The export is arranged exactly for `reasongraph`'s
`OnnxTokenClassifierExtractor`. For a model directory `<dir>`:

- `<dir>/config.json` with `id2label` (written by training)
- tokenizer files in `<dir>` (written by training)
- `<dir>/onnx/model.onnx`, inputs `input_ids` + `attention_mask`, output
  per-token logits `[seq_len, num_labels]`

`causal-export-onnx <dir> --check` runs one `onnxruntime` inference with those
exact feeds and asserts the output shape before you ship the model.

`OnnxTokenClassifierExtractor.__call__` returns a flat entity list and discards
the span TYPE, so `reasongraph` needs a thin `CausalOnnxExtractor` wrapper that
keeps the types and pairs cause->effect. `evaluate.py`'s `decode_bio_spans` /
`to_causal_relations` are the reference implementation of that decode.

## Tests

```bash
pytest
```

The tests cover the model-free core (CNC tag parsing incl. nested tags, sentinel
wrap/project and its drop-on-failure validation). One guard test loads the real
NLLB tokenizer to assert the sentinels are not `<unk>` (skips if unavailable);
the rest download nothing.

## Running the training

Training and export pull in torch / transformers / optimum and run on your own
infrastructure (CPU works; a single GPU is far faster). On a Blackwell GPU
(RTX 50-series, sm_120) use an isolated venv with a cu128 torch build -- older
torch reports the GPU as available but fails at kernel launch. Pin torch and
install the rest under a constraints file so nothing swaps the cu128 build for a
default-CUDA wheel (which mixes nvidia cu12/cu13 libs and breaks NCCL at import).
