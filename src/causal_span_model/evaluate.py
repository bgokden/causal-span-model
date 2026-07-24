"""Evaluate a trained causal span model and decode directed causal relations.

Two jobs:

  * ``report`` -- span-level seqeval precision/recall/F1, broken down PER LANGUAGE
    (the model is multilingual, so a single aggregate F1 hides per-language
    quality) plus the aggregate.
  * ``decode_bio_spans`` / ``to_causal_relations`` -- turn per-token BIO tags into
    typed spans and then directed ``{"cause", "effect"}`` pairs. This mirrors what
    reasongraph's planned ``CausalOnnxExtractor`` wrapper must do on top of the
    generic ``OnnxTokenClassifierExtractor`` (whose ``__call__`` returns a flat
    entity list and discards the span TYPE). Keeping the reference decode here
    documents the contract and lets you eyeball model output as cause->effect.
"""

import argparse
import sys

from causal_span_model.labels import ID2LABEL


def decode_bio_spans(tokens: list[str], tags: list[str]) -> dict[str, list[str]]:
    """Group BIO tags into ``{"CAUSE": [...], "EFFECT": [...], "SIGNAL": [...]}``.

    A ``B-`` tag or a type change starts a new span; ``I-`` of the same type
    continues it. Mirrors reasongraph ``OnnxTokenClassifierExtractor._decode_bio``.
    """
    spans: dict[str, list[str]] = {"CAUSE": [], "EFFECT": [], "SIGNAL": []}
    current_type: str | None = None
    current: list[str] = []

    def flush() -> None:
        nonlocal current_type, current
        if current_type is not None and current:
            spans[current_type].append(" ".join(current))
        current_type, current = None, []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush()
            continue
        prefix, span_type = tag.split("-", 1)
        if prefix == "B" or span_type != current_type:
            flush()
            current_type, current = span_type, [token]
        else:
            current.append(token)
    flush()
    return spans


def to_causal_relations(tokens: list[str], tags: list[str]) -> list[dict[str, str]]:
    """Pair decoded cause and effect spans into directed relations.

    Direction comes straight from the label types, so reversed phrasing
    ("the crash resulted from brake failure") is handled by the model rather than
    a cue lexicon. This is a REFERENCE decoder with a deliberate simplification:
    it pairs the i-th cause with the i-th effect positionally and assumes roughly
    one cause / one effect per sentence. Multi-relation sentences and
    signal-split argument chunks are not merged; a production consumer may want a
    smarter pairing.
    """
    spans = decode_bio_spans(tokens, tags)
    causes = spans["CAUSE"]
    effects = spans["EFFECT"]
    relations: list[dict[str, str]] = []
    for index in range(min(len(causes), len(effects))):
        relations.append({"cause": causes[index], "effect": effects[index]})
    return relations


def token_level_prf(gold_all: list[list[str]], pred_all: list[list[str]]) -> tuple[float, float, float]:
    """Micro precision/recall/F1 over token TYPES (CAUSE/EFFECT/SIGNAL vs O).

    Complements seqeval's exact-span F1, which is brutal on CNC's long clause
    spans (a one-token boundary miss zeroes the whole span). This counts a token
    correct when its collapsed type matches, so it reflects how much of the causal
    content the model localizes regardless of exact boundaries.
    """
    tp = fp = fn = 0
    for gold_row, pred_row in zip(gold_all, pred_all):
        for gold, pred in zip(gold_row, pred_row):
            gold_type = gold.split("-", 1)[1] if gold != "O" else "O"
            pred_type = pred.split("-", 1)[1] if pred != "O" else "O"
            if gold_type != "O" and gold_type == pred_type:
                tp += 1
            else:
                if pred_type != "O":
                    fp += 1
                if gold_type != "O":
                    fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def report(model_dir: str, eval_path: str, max_len: int = 256, show: int = 5) -> None:
    """Print per-language and aggregate seqeval reports plus a few decoded relations."""
    from seqeval.metrics import classification_report, f1_score
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from causal_span_model.dataset import drop_overlong, load_bio_jsonl

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    dataset = drop_overlong(load_bio_jsonl(eval_path), tokenizer, max_len)
    by_lang: dict[str, tuple[list, list]] = {}
    gold_all: list[list[str]] = []
    pred_all: list[list[str]] = []
    shown = 0
    mismatches = 0
    for example in dataset:
        tokens = example["tokens"]
        gold = [ID2LABEL[label_id] for label_id in example["labels"]]
        pred = _predict(model, tokenizer, tokens, max_len)
        if len(pred) != len(gold):
            mismatches += 1
            size = min(len(pred), len(gold))
            gold, pred = gold[:size], pred[:size]
        lang = example.get("lang", "en")
        bucket = by_lang.setdefault(lang, ([], []))
        bucket[0].append(gold)
        bucket[1].append(pred)
        gold_all.append(gold)
        pred_all.append(pred)
        if shown < show:
            print(f"text: {' '.join(tokens)}")
            print(f"  relations: {to_causal_relations(tokens, pred)}")
            shown += 1

    if mismatches:
        print(f"[eval] WARNING: {mismatches} examples had pred/gold length mismatch "
              f"(aligned to the shorter)")
    for lang in sorted(by_lang):
        gold_lang, pred_lang = by_lang[lang]
        print(f"\n=== lang={lang} (n={len(gold_lang)}) F1={f1_score(gold_lang, pred_lang):.4f} ===")
        print(classification_report(gold_lang, pred_lang))
    print(f"\n=== AGGREGATE exact-span seqeval (n={len(gold_all)}) ===")
    print(classification_report(gold_all, pred_all))
    tp, tr, tf = token_level_prf(gold_all, pred_all)
    print(f"=== AGGREGATE token-level (type match, boundary-tolerant) ===")
    print(f"precision {tp:.4f}  recall {tr:.4f}  f1 {tf:.4f}")


def _predict(model, tokenizer, tokens: list[str], max_len: int) -> list[str]:
    import torch

    encoded = tokenizer(
        tokens, is_split_into_words=True, truncation=True,
        max_length=max_len, return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits[0]
    tag_ids = logits.argmax(-1).tolist()
    word_ids = encoded.word_ids(batch_index=0)
    tags: list[str] = []
    previous = None
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous:
            previous = word_id
            continue
        tags.append(ID2LABEL[int(tag_ids[position])])
        previous = word_id
    return tags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="trained checkpoint directory")
    parser.add_argument("eval_path", help="evaluation BIO JSONL")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--show", type=int, default=5,
                        help="how many decoded relations to print")
    args = parser.parse_args(argv)

    report(args.model_dir, args.eval_path, max_len=args.max_len, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
