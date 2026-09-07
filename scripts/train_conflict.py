"""Train a cross-encoder conflict classifier on (old, new) sentence pairs.

Given pairs labelled conflict (1) / no-conflict (0), fine-tune a binary
sequence-classification head on mDeBERTa-v3-base. Pairs are split by the base fact
(the ``old`` text) so paraphrases/updates of the same seed never straddle train and
test. Reports accuracy / precision / recall / F1 on the held-out split and, optionally,
on an external pair set (e.g. reasongraph ``tests/data/contradictions.jsonl``).

Run:
    python scripts/train_conflict.py --pairs data/conflicts/pairs.jsonl \
        --ext ~/repos/reasongraph/tests/data/contradictions.jsonl --output-dir outputs/conflict
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


def _to_label(value) -> int:
    return 1 if value is True or str(value).strip().lower() in ("1", "true", "yes") else 0


def load_pairs(path: str) -> list[tuple[str, str, int]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append((str(r["old"]), str(r["new"]), _to_label(r["label"])))
    return rows


def split_by_fact(rows, test_frac=0.15):
    train, test = [], []
    for old, new, y in rows:
        bucket = int(hashlib.md5(old.encode("utf-8")).hexdigest(), 16) % 100
        (test if bucket < test_frac * 100 else train).append((old, new, y))
    return train, test


def _encode(tok, batch, max_len, device):
    enc = tok([b[0] for b in batch], [b[1] for b in batch],
              truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    labels = torch.tensor([b[2] for b in batch])
    return {k: v.to(device) for k, v in enc.items()}, labels.to(device)


def evaluate(model, tok, rows, max_len, device, batch_size=32) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            enc, y = _encode(tok, rows[i:i + batch_size], max_len, device)
            pred = model(**enc).logits.argmax(-1)
            for p, t in zip(pred.tolist(), y.tolist()):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 1:
                    fn += 1
                else:
                    tn += 1
    total = max(1, tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"acc": (tp + tn) / total, "precision": precision, "recall": recall,
            "f1": f1, "n": len(rows)}


def train(pairs_csv, ext_path, base_model, output_dir, epochs, lr, batch_size, max_len, seed):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2).to(device)

    rows = load_pairs(pairs_csv)
    train_rows, test_rows = split_by_fact(rows)
    pos = sum(1 for _, _, y in rows if y == 1)
    print(f"[conflict-train] pairs {len(rows)} (pos {pos}) -> train {len(train_rows)} "
          f"test {len(test_rows)} device {device}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(train_rows) + batch_size - 1) // batch_size * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        for i in range(0, len(order), batch_size):
            batch = [train_rows[j] for j in order[i:i + batch_size]]
            enc, y = _encode(tok, batch, max_len, device)
            loss = model(**enc, labels=y).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        m = evaluate(model, tok, test_rows, max_len, device)
        print(f"[conflict-train] epoch {epoch + 1}/{epochs} test acc {m['acc']:.3f} f1 {m['f1']:.3f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    m = evaluate(model, tok, test_rows, max_len, device)
    print(f"[conflict-train] FINAL held-out: acc {m['acc']:.3f} precision {m['precision']:.3f} "
          f"recall {m['recall']:.3f} f1 {m['f1']:.3f} (n={m['n']})")
    if ext_path and os.path.exists(ext_path):
        e = evaluate(model, tok, load_pairs(ext_path), max_len, device)
        print(f"[conflict-train] EXTERNAL {ext_path}: acc {e['acc']:.3f} precision {e['precision']:.3f} "
              f"recall {e['recall']:.3f} f1 {e['f1']:.3f} (n={e['n']})")
    print(f"[conflict-train] saved -> {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/conflicts/pairs.jsonl")
    parser.add_argument("--ext", default=None, help="external eval jsonl (old,new,label)")
    parser.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--output-dir", default="outputs/conflict")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    train(args.pairs, args.ext, args.base_model, args.output_dir, args.epochs, args.lr,
          args.batch_size, args.max_len, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
