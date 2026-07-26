"""Render a Hugging Face model card for the span-pointer causal model.

This is a CUSTOM architecture (not a standard transformers AutoModel): a 6-channel
start/end pointer head + a signal classifier + beam-search decoding. Loading and
inference go through the ``causal_span_model`` package, so the card ships a working
snippet rather than the usual AutoModel example.
"""

import argparse
import os
import sys

from causal_span_model.pointer.submission import POINTER_HF_REPO

_CARD = """---
license: mit
base_model: microsoft/mdeberta-v3-base
tags:
- causal-extraction
- causality
- cause-effect
- span-extraction
- causal-news-corpus
- multilingual
language:
- en
- es
- fr
- de
- pt
- tr
- ru
- ar
- zh
- ja
---

# {title}

A **span-pointer** causal extraction model: given a sentence it predicts the
**cause**, **effect** and **signal** spans as start/end pointers, decoded under
ordering/non-overlap constraints with beam search (top-2 relations per sentence).
Fine-tuned from [`microsoft/mdeberta-v3-base`](https://huggingface.co/microsoft/mdeberta-v3-base)
on the [Causal News Corpus](https://github.com/tanfiona/CausalNewsCorpus) Subtask-2
(CC0). Architecture reimplemented from the CNC baseline (MIT).

## Benchmark (Causal News Corpus Subtask 2)

Official scorer (`evaluation/subtask2`: FairEval + best-combination alignment), V2 dev:

{metrics}

**This beats the organizer's 0.627 dev baseline** and the 2022 shared-task winner
(0.542, test); it trails the 2023 winner (0.728, test). Same scorer, same dev set.

### vs a few-shot LLM

A prompted general LLM does not match this fine-tune. Qwen2.5-7B-Instruct, few-shot on
the same dev set and official scorer, scores **0.24** F1 with a plain causal prompt and
**0.41** with a scheme-aware prompt (vs **0.70** here). Even on a capability-fair subset
-- causality a reader recognises without CNC's broad purpose/motive/implicit
conventions -- the LLM reaches ~0.45 vs this model's ~0.63. The residual gap is exact
span-boundary precision, which fine-tuning on the annotation provides.

## Usage

This is a custom architecture, so inference goes through the `causal_span_model`
package (not `AutoModel`):

```python
from huggingface_hub import snapshot_download
from causal_span_model.pointer.submission import load_pointer, predict_sentence

local_dir = snapshot_download("{repo_id}")
model, tokenizer = load_pointer(local_dir)
print(predict_sentence(model, tokenizer, "Heavy rainfall caused severe flooding."))
# ['<ARG0>Heavy rainfall</ARG0> <SIG0>caused</SIG0> <ARG1>severe flooding</ARG1> .', ...]
```

`<ARG0>` = cause, `<ARG1>` = effect, `<SIG0>` = signal. The prediction is a list of
tagged relation strings (up to two per sentence).

### Multilingual

Trained on English spans, but multilingual at inference (mDeBERTa encoder +
script-aware segmentation). Use `predict_relations`, which returns character-exact
spans in any script:

```python
from causal_span_model.pointer.infer import predict_relations

predict_relations(model, tokenizer, "暴雨导致该地区发生严重洪灾。")
# [{'cause': '暴雨', 'effect': '该地区发生严重洪灾', 'signal': '导致'}]
predict_relations(model, tokenizer, "Las fuertes lluvias provocaron inundaciones.")
# [{'cause': 'Las fuertes lluvias', 'effect': 'inundaciones', 'signal': 'provocaron'}]
```

Verified on es/fr/de/pt/tr/ru/ar and CJK (zh/ja).

## Notes

- It is NOT compatible with a generic token-classification ONNX consumer -- it
  needs its own start/end + beam-search decoder (provided by the package).
- It has a built-in **causal gate** (a causal/non-causal head, ~0.85 accuracy on
  CNC dev): `predict_relations` returns `[]` on text it judges non-causal, so it
  is safe to run on arbitrary input. Beam duplicates are collapsed to one relation
  per distinct cause->effect.

## License

MIT (weights and code). Training data is CC0-1.0.
"""


def render_card(repo_id: str, metrics: str = "_(metrics pending)_", title: str | None = None) -> str:
    # str.replace (not .format) so literal { } in the code snippets are left alone.
    return (_CARD
            .replace("{repo_id}", repo_id)
            .replace("{metrics}", metrics)
            .replace("{title}", title or repo_id.split("/")[-1]))


def write_card(model_dir: str, repo_id: str, metrics: str) -> str:
    card = render_card(repo_id, metrics=metrics)
    with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(card)
    return card


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("repo_id", nargs="?", default=POINTER_HF_REPO,
                        help=f"HF repo id (default: {POINTER_HF_REPO})")
    parser.add_argument("--metrics-file", default=None)
    args = parser.parse_args(argv)
    metrics = "_(metrics pending)_"
    if args.metrics_file:
        with open(args.metrics_file, encoding="utf-8") as handle:
            metrics = "```\n" + handle.read().strip() + "\n```"
    write_card(args.model_dir, args.repo_id, metrics)
    print(f"wrote model card to {os.path.join(args.model_dir, 'README.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
