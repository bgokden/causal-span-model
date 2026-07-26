"""Generate a CNC Subtask-2 submission.json from a trained pointer model.

Runs the model on each grouped-dev sentence, beam-decodes the top-2 relations,
projects subword span indices back to word positions, wraps the words in CNC tags,
and writes one JSON line per sentence (a list of tagged relation strings). Feed the
output to the official scorer (cnc_eval/_evaluate.py).
"""

import argparse
import json
import os
import sys

import torch

from causal_span_model.pointer.decode import decode_relations
from causal_span_model.pointer.model import PointerCausalModel

# Canonical Hugging Face repo for the published pointer model. Single source of
# truth: the push target, the model card, and downstream consumers (reasongraph's
# CausalPointerExtractor default) all reference this exact id so they cannot drift.
# Distinct from the BIO tagger's repo (causal-span-mdeberta).
POINTER_HF_REPO = "berk/causal-span-pointer-mdeberta"


def load_pointer(model_dir: str):
    from transformers import AutoTokenizer

    with open(os.path.join(model_dir, "pointer_config.json")) as handle:
        cfg = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = PointerCausalModel(cfg["base_model"], cfg.get("dropout", 0.1))
    state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state)
    # Checkpoints are saved in half precision (bf16 training). Cast to fp32 so
    # inference runs consistently on CPU, where mixing half weights with the
    # fp32-loaded base encoder raises "mat1 and mat2 must have the same dtype".
    model.float()
    model.eval()
    return model, tokenizer


def _wrap(words, word_ids, span, open_tag, close_tag):
    start_tok, end_tok = span
    start_word, end_word = word_ids[start_tok], word_ids[end_tok]
    if start_word is None or end_word is None:
        return
    words[start_word] = open_tag + words[start_word]
    words[end_word] = words[end_word] + close_tag


def _tagged(words, word_ids, relation):
    out = list(words)
    _wrap(out, word_ids, relation["cause"], "<ARG0>", "</ARG0>")
    _wrap(out, word_ids, relation["effect"], "<ARG1>", "</ARG1>")
    if relation["signal"] is not None:
        _wrap(out, word_ids, relation["signal"], "<SIG0>", "</SIG0>")
    return " ".join(out)


def predict_sentence(model, tokenizer, text, max_len=256, topk=5, device="cpu"):
    words = str(text).split(" ")
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device))
    length = enc["input_ids"].shape[1]
    logits = {k: out[k][0].cpu() for k in
              ("cause_start", "cause_end", "effect_start", "effect_end", "sig_start", "sig_end")}
    has_signal = bool(out["signal_cls"][0].argmax().item())
    relations = decode_relations(logits, sep_pos=length - 1, has_signal=has_signal,
                                 beam=True, topk=topk)
    predictions = [_tagged(words, word_ids, rel) for rel in relations]
    if len(predictions) == 1:  # ensure two candidates for multi-relation coverage
        predictions = predictions * 2
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("ref_csv", help="grouped reference csv (text + rows in order)")
    parser.add_argument("out_json")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args(argv)

    import pandas as pd  # CLI-only (submission scoring); kept out of the inference import path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_pointer(args.model_dir)
    model.to(device)
    df = pd.read_csv(args.ref_csv)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        for position, (_, row) in enumerate(df.iterrows()):
            preds = predict_sentence(model, tokenizer, row["text"], args.max_len, args.topk, device)
            handle.write(json.dumps({"index": position, "prediction": preds}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out_json} ({len(df)} sentences)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
