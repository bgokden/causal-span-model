"""Embedding-based causal / non-causal gate.

Trains a logistic-regression and a 2-layer MLP on frozen sentence-embeddings to decide whether
a sentence expresses a causal relation. This is decoupled from the span-pointer model: run the
pointer with its own gate OFF and let this classifier decide, so the CNC span benchmark is
unaffected by construction.

Data: CNC subtask-1 (label 0/1), synthetic positives + negatives (6 langs), R7 general-prose
negatives. Held out: CNC-1 dev + synthetic dev. Reports accuracy / F1 overall and per
language / domain, and saves each classifier as a .joblib.

Run:
    python scripts/train_embed_gate.py --out-dir outputs/embed_gate
"""

import argparse
import collections
import csv
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def build_rows():
    train, dev = [], []
    for r in _rows("data/cnc/train_subtask1.csv"):
        train.append((str(r["text"]), int(r["label"]), "en", "news"))
    for r in _rows("data/cnc/dev_subtask1.csv"):
        dev.append((str(r["text"]), int(r["label"]), "en", "news"))
    for r in _rows("data/prepared/train.csv"):
        train.append((str(r["text"]), 1, r.get("lang", "en"), r.get("domain", "?")))
    for r in _rows("data/prepared/negatives.csv"):
        train.append((str(r["text"]), 0, r.get("lang", "en"), r.get("domain", "?")))
    for r in _rows("data/prepared/prose_negatives.csv"):
        train.append((str(r["text"]), 0, r.get("lang", "en"), "prose"))
    for r in _rows("data/prepared/dev.csv"):
        dev.append((str(r["text"]), 1 if int(r["num_rs"]) > 0 else 0, r.get("lang", "en"), r.get("domain", "?")))
    return train, dev


def report(tag, y, pred, meta):
    print(f"  {tag}: acc {accuracy_score(y, pred):.3f} f1 {f1_score(y, pred, zero_division=0):.3f} (n={len(y)})")
    for key, idx in (("lang", 0), ("domain", 1)):
        groups = collections.defaultdict(lambda: ([], []))
        for yi, pi, m in zip(y, pred, meta):
            g = m[idx]
            groups[g][0].append(yi)
            groups[g][1].append(pi)
        cells = " ".join(f"{g}:{f1_score(gy, gp, zero_division=0):.2f}"
                         for g, (gy, gp) in sorted(groups.items()))
        print(f"    by {key}: {cells}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/embed_gate")
    parser.add_argument("--models", nargs="*", default=[
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ])
    args = parser.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    train, dev = build_rows()
    xtr = [r[0] for r in train]
    ytr = np.array([r[1] for r in train])
    xdv = [r[0] for r in dev]
    ydv = np.array([r[1] for r in dev])
    dv_meta = [(r[2], r[3]) for r in dev]
    print(f"train {len(train)} (pos {int(ytr.sum())}) | dev {len(dev)} (pos {int(ydv.sum())})")
    os.makedirs(args.out_dir, exist_ok=True)

    best = None
    for name in args.models:
        st = SentenceTransformer(name, device="cuda")
        etr = st.encode(xtr, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        edv = st.encode(xdv, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        short = name.split("/")[-1]
        print(f"[{short}]")
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(etr, ytr)
        report("LR ", ydv, lr.predict(edv), dv_meta)
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True,
                            random_state=42).fit(etr, ytr)
        report("MLP", ydv, mlp.predict(edv), dv_meta)
        for clf, kind in ((lr, "lr"), (mlp, "mlp")):
            f1 = f1_score(ydv, clf.predict(edv), zero_division=0)
            path = os.path.join(args.out_dir, f"gate_{short}_{kind}.joblib")
            joblib.dump({"embed_model": name, "kind": kind, "clf": clf}, path)
            if best is None or f1 > best[0]:
                best = (f1, path)
    print(f"best held-out F1 {best[0]:.3f} -> {best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
