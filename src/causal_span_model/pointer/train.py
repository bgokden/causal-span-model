"""Train the span-pointer causal model and select the best epoch on a fast proxy.

The official CNC scorer (FairEval + best-combination) is too slow to run every
epoch, so model selection uses a fast exact-span F1 proxy (greedy decode of
cause/effect/signal vs gold spans on dev). The slow official score is run once at
the end via ``pointer.submission``.
"""

import argparse
import json
import os
import sys

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from causal_span_model.pointer.data import (
    build_dataset,
    build_negatives,
    non_causal_texts,
    relations_from_csv,
)
from causal_span_model.pointer.decode import _prep, decode_cause_effect_greedy, decode_signal
from causal_span_model.pointer.model import PointerCausalModel


def _collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention, starts, ends, causal = [], [], [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attention.append(x["attention_mask"] + [0] * pad)
        starts.append(x["start_positions"])
        ends.append(x["end_positions"])
        causal.append(x["causal_labels"])
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention),
        "start_positions": torch.tensor(starts),
        "end_positions": torch.tensor(ends),
        "causal_labels": torch.tensor(causal),
    }


def _fast_eval(model, examples, device) -> float:
    """Exact-span micro F1 over cause/effect/signal via greedy decode (proxy)."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x in examples:
            ids = torch.tensor([x["input_ids"]], device=device)
            mask = torch.tensor([x["attention_mask"]], device=device)
            out = model(input_ids=ids, attention_mask=mask)
            length = len(x["input_ids"])
            logits = {k: out[k][0] for k in
                      ("cause_start", "cause_end", "effect_start", "effect_end", "sig_start", "sig_end")}
            cs, ce = _prep(logits["cause_start"], length - 1), _prep(logits["cause_end"], length - 1)
            es, ee = _prep(logits["effect_start"], length - 1), _prep(logits["effect_end"], length - 1)
            (sc, ec, se_, ee_), = decode_cause_effect_greedy(cs, ce, es, ee)
            gold_sp, gold_ep = x["start_positions"], x["end_positions"]
            preds = [(sc, ec), (se_, ee_)]
            golds = [(gold_sp[0], gold_ep[0]), (gold_sp[1], gold_ep[1])]
            if gold_ep[2] != -100:
                ss, es2 = decode_signal(logits["sig_start"], logits["sig_end"], length - 1)
                preds.append((ss, es2))
                golds.append((gold_sp[2], gold_ep[2]))
            for pred, gold in zip(preds, golds):
                if pred == gold:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _save(model, tokenizer, output_dir, base_model, dropout):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "pointer_config.json"), "w") as handle:
        json.dump({"base_model": base_model, "dropout": dropout}, handle)


def train(train_csv, dev_csv, output_dir, base_model="microsoft/mdeberta-v3-base",
          epochs=10, lr=3e-5, batch_size=16, max_len=256, seed=42,
          warmup_ratio=0.06, dropout=0.1, extra_csvs=None, neg_csv=None):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    pad_id = tokenizer.pad_token_id

    train_rels = relations_from_csv(train_csv)
    for extra in extra_csvs or []:
        train_rels.extend(relations_from_csv(extra))
    train_ex = list(build_dataset(train_rels, tokenizer, max_len))
    negatives = 0
    if neg_csv:
        neg_ex = list(build_negatives(non_causal_texts(neg_csv), tokenizer, max_len))
        train_ex = train_ex + neg_ex
        negatives = len(neg_ex)
    dev_ex = list(build_dataset(relations_from_csv(dev_csv), tokenizer, max_len))
    print(f"[pointer-train] train={len(train_ex)} (negatives={negatives}) "
          f"dev={len(dev_ex)} device={device}")

    model = PointerCausalModel(base_model, dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(train_ex) + batch_size - 1) // batch_size * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * warmup_ratio), steps)

    order = list(range(len(train_ex)))
    generator = torch.Generator().manual_seed(seed)
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        for batch_idx in _batches(order, batch_size, generator):
            batch = _collate([train_ex[j] for j in batch_idx], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        f1 = _fast_eval(model, dev_ex, device)
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            _save(model, tokenizer, output_dir, base_model, dropout)
            marker = " (best, saved)"
        print(f"[pointer-train] epoch {epoch + 1}/{epochs} dev exact-span F1 {f1:.4f}{marker}")
    print(f"[pointer-train] best dev proxy F1 {best_f1:.4f} -> {output_dir}")


def _batches(order, batch_size, generator):
    shuffled = [order[k] for k in torch.randperm(len(order), generator=generator).tolist()]
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra", nargs="*", default=None,
                        help="extra training CSVs (e.g. CNC augmented data)")
    parser.add_argument("--negatives", default=None,
                        help="grouped CSV supplying non-causal sentences for the causal gate")
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train(args.train, args.dev, args.output_dir, base_model=args.base_model,
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
          max_len=args.max_len, seed=args.seed, extra_csvs=args.extra,
          neg_csv=args.negatives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
