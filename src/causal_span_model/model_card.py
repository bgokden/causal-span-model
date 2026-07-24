"""Render a Hugging Face model card (README.md) for the trained causal span model.

The card carries the HF metadata frontmatter (license, base_model, tags, pipeline)
plus usage, evaluation and limitations. Metrics are injected after training so the
card reflects the real numbers.
"""

import argparse
import os
import sys

_CARD = """---
license: mit
base_model: {base_model}
pipeline_tag: token-classification
library_name: transformers
tags:
- token-classification
- causal-extraction
- causality
- cause-effect
- multilingual
- reasongraph
language:
- multilingual
---

# {title}

A multilingual BIO token classifier that tags **cause**, **effect** and **signal**
spans, fine-tuned from [`{base_model}`](https://huggingface.co/{base_model}) on the
[Causal News Corpus](https://github.com/tanfiona/CausalNewsCorpus) (CC0-1.0).
Direction is encoded in the label TYPE, so `cause -> effect` is read straight from
the tags -- reversed phrasing like "the crash resulted from brake failure" is
handled without a separate orientation lexicon.

Built for [reasongraph](https://github.com/bgokden/reasongraph) as the accurate,
model-based causal extractor.

## Labels

```
O
B-CAUSE  I-CAUSE     the cause span
B-EFFECT I-EFFECT    the effect span
B-SIGNAL I-SIGNAL    the causal connective
```

## Usage (transformers)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForTokenClassification.from_pretrained("{repo_id}").eval()

enc = tok("Heavy rainfall caused severe flooding.", return_tensors="pt")
ids = model(**enc).logits[0].argmax(-1).tolist()
toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
print([(t, model.config.id2label[i]) for t, i in zip(toks, ids)])
```

An ONNX build (`onnx/model.onnx`, inputs `input_ids` + `attention_mask`, output
per-token logits) is included for fast CPU inference via onnxruntime -- this is
what reasongraph loads.

## Evaluation

Benchmarked on the Causal News Corpus (CNC) Subtask 2 dev set. This is an honest
**first-pass** model: it is **below the shared-task baseline and SOTA**. Its value
is robust multilingual zero-shot extraction, not leaderboard rank.

{metrics}

Multilingual capability comes from mDeBERTa-v3's zero-shot cross-lingual transfer;
the training spans are English only.

## Training data

Causal News Corpus V2, subtask 2 (CC0-1.0). ARG0 = cause, ARG1 = effect,
SIG* = signal. Tags can nest (a signal inside an argument) and a sentence may carry
several relations.

## Limitations

- Trained on English news text; other languages rely on zero-shot transfer and are
  less accurate, especially for distant scripts (e.g. Chinese, Arabic).
- News-domain bias; short, explicit causal statements are handled best.
- Multi-relation sentences receive one predicted causal structure.

## License

MIT (weights and code). Training data is CC0-1.0.
"""


def render_card(
    repo_id: str,
    base_model: str = "microsoft/mdeberta-v3-base",
    metrics: str = "_(metrics pending)_",
    title: str | None = None,
) -> str:
    return _CARD.format(
        repo_id=repo_id,
        base_model=base_model,
        metrics=metrics,
        title=title or repo_id.split("/")[-1],
    )


def write_card(model_dir: str, repo_id: str, **kwargs) -> str:
    card = render_card(repo_id, **kwargs)
    with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(card)
    return card


def _read_metrics(path: str | None) -> str:
    if not path:
        return "_(metrics pending)_"
    with open(path, encoding="utf-8") as handle:
        return "```\n" + handle.read().strip() + "\n```"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("repo_id", help="e.g. berkgokden/causal-span-mdeberta")
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--metrics-file", default=None,
                        help="text file whose contents become the Evaluation block")
    args = parser.parse_args(argv)

    write_card(args.model_dir, args.repo_id,
               base_model=args.base_model, metrics=_read_metrics(args.metrics_file))
    print(f"wrote model card to {os.path.join(args.model_dir, 'README.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
