"""Reference raw-text inference: predict typed causal spans from a sentence.

Mirrors what reasongraph's planned ``CausalOnnxExtractor`` does on top of the ONNX
model: tokenize, argmax per token, group BIO tags, and slice CHARACTER spans out of
the source text via ``offset_mapping``. Direction comes from the label type, so
cause/effect are recovered directly. Use it to eyeball the model, including
zero-shot behaviour on non-English text.

``--demo`` runs a fixed multilingual set (the same causal sentence across
languages) so the cross-lingual transfer can be inspected at a glance.
"""

import argparse
import sys

# One causal statement ("heavy rain -> flooding") across the target languages, to
# eyeball zero-shot transfer from the English-only training data.
_DEMO = [
    ("en", "Heavy rainfall caused severe flooding in the region."),
    ("es", "Las fuertes lluvias provocaron graves inundaciones en la región."),
    ("fr", "De fortes pluies ont provoqué de graves inondations dans la région."),
    ("de", "Starke Regenfälle verursachten schwere Überschwemmungen in der Region."),
    ("pt", "As fortes chuvas causaram graves inundações na região."),
    ("tr", "Şiddetli yağışlar bölgede ciddi sellere neden oldu."),
    ("ru", "Сильные дожди вызвали серьёзные наводнения в регионе."),
    ("zh", "暴雨导致该地区发生严重洪灾。"),
    ("ar", "تسببت الأمطار الغزيرة في فيضانات شديدة في المنطقة."),
]


def load(model_dir: str):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    return model, tokenizer


def predict(model, tokenizer, text: str, max_len: int = 256) -> dict:
    """Return ``{"spans": {...}, "relations": [...]}`` for one raw sentence."""
    import torch

    enc = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        tag_ids = model(**enc).logits[0].argmax(-1).tolist()

    spans: dict[str, list[str]] = {"CAUSE": [], "EFFECT": [], "SIGNAL": []}
    current = {"type": None, "start": 0, "end": 0}

    def flush() -> None:
        if current["type"] is not None:
            fragment = text[current["start"]:current["end"]].strip()
            if fragment:
                spans[current["type"]].append(fragment)
        current["type"] = None

    for (start, end), tag_id in zip(offsets, tag_ids):
        if start == end:  # special token
            continue
        label = model.config.id2label[tag_id]
        if label == "O":
            flush()
            continue
        prefix, span_type = label.split("-", 1)
        if prefix == "B" or span_type != current["type"]:
            flush()
            current.update(type=span_type, start=start, end=end)
        else:
            current["end"] = end
    flush()

    causes, effects = spans["CAUSE"], spans["EFFECT"]
    relations = [
        {"cause": causes[i], "effect": effects[i]}
        for i in range(min(len(causes), len(effects)))
    ]
    return {"spans": spans, "relations": relations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--demo", action="store_true",
                        help="run the built-in multilingual examples")
    args = parser.parse_args(argv)

    model, tokenizer = load(args.model_dir)
    items = _DEMO if args.demo else [("?", args.text)]
    for lang, text in items:
        if not text:
            continue
        out = predict(model, tokenizer, text)
        print(f"[{lang}] {text}")
        print(f"      spans={out['spans']}")
        print(f"      relations={out['relations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
