"""H4: conflict pre-filter trained for RECALL.

A cheap embedding-pair classifier that runs BEFORE the LLM conflict resolver (H3) to cut the
candidate pairs it must check. Trained class-weighted for high recall so it rarely drops a
real conflict, then a threshold sweep finds the operating point that filters most
non-conflicts while keeping recall >= 0.98.

Same features as G2: [a, b, |a-b|, a*b] over `paraphrase-multilingual-MiniLM-L12-v2`.
Train: 80% of K1 pairs + the 200 LLM paraphrases. Test: the held-out 20% of K1 (clean) and
the 40 hand pairs (note: the paraphrases are derived from the 40 hand pairs, so the hand-pair
scores are optimistic; the K1 held-out split is the clean read).

Saves `{embed_model, kind, clf, threshold, features}` to --out (default the lab models dir),
the same joblib shape G2/rg already load.

Run:
    cd ~/repos/causal-span-model
    uv run python scripts/train_conflict_prefilter.py
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
K1 = "/home/berk/repos/primaxiom-lab/causal/data/conflicts/pairs.jsonl"
PARAPHRASES = "/home/berk/repos/causal-span-model/data/llm/conflict_paraphrases.jsonl"
HAND = "/home/berk/repos/reasongraph/tests/data/contradictions.jsonl"
OUT = "/home/berk/repos/primaxiom-lab/causal/models/conflict_prefilter_lr.joblib"


def _jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def feats(st, rows):
    a = st.encode([str(r["old"]) for r in rows], batch_size=256,
                  normalize_embeddings=True, show_progress_bar=False)
    b = st.encode([str(r["new"]) for r in rows], batch_size=256,
                  normalize_embeddings=True, show_progress_bar=False)
    x = np.concatenate([a, b, np.abs(a - b), a * b], axis=1)
    y = np.array([int(bool(r["label"])) for r in rows])
    return x, y


def report(name, y, proba, thresholds):
    print(f"\n  [{name}] n={len(y)} (conflict {int(y.sum())} / non {int((1 - y).sum())})")
    print(f"    {'thr':>5s} {'recall':>7s} {'precision':>10s} {'TNR(cut non)':>13s} "
          f"{'filtered-all':>12s}")
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rec = recall_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        neg = y == 0
        tnr = float((pred[neg] == 0).mean()) if neg.any() else 0.0  # non-conflicts filtered
        filtered = float((pred == 0).mean())                        # all pairs filtered
        rows.append((t, rec, prec, tnr, filtered))
        print(f"    {t:>5.2f} {rec:>7.3f} {prec:>10.3f} {tnr:>13.3f} {filtered:>12.3f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weight", default="balanced",
                    help="'balanced' or a weight for the conflict class, e.g. 3")
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    k1 = _jsonl(K1)
    para = _jsonl(PARAPHRASES)
    hand = _jsonl(HAND)

    # stratified K1 train/held-out split
    import random
    rng = random.Random(args.seed)
    pos = [r for r in k1 if int(bool(r["label"]))]
    neg = [r for r in k1 if not int(bool(r["label"]))]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_te = int(len(pos) * args.test_frac)
    n_neg_te = int(len(neg) * args.test_frac)
    k1_test = pos[:n_pos_te] + neg[:n_neg_te]
    k1_train = pos[n_pos_te:] + neg[n_neg_te:]
    rng.shuffle(k1_train)
    train_rows = k1_train + para
    print(f"train {len(train_rows)} (K1-train {len(k1_train)} + paraphrases {len(para)}) | "
          f"K1 held-out {len(k1_test)} | hand {len(hand)}", flush=True)

    st = SentenceTransformer(EMBED, device="cuda")
    xtr, ytr = feats(st, train_rows)
    xte_k1, yte_k1 = feats(st, k1_test)
    xte_hand, yte_hand = feats(st, hand)

    try:
        cw = {0: 1.0, 1: float(args.class_weight)}
    except ValueError:
        cw = args.class_weight  # "balanced"
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight=cw).fit(xtr, ytr)
    causal_col = list(clf.classes_).index(1)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    p_k1 = clf.predict_proba(xte_k1)[:, causal_col]
    p_hand = clf.predict_proba(xte_hand)[:, causal_col]
    k1_rows = report("K1 held-out (clean)", yte_k1, p_k1, thresholds)
    report("40 hand pairs (paraphrase leakage -> optimistic)", yte_hand, p_hand, thresholds)

    # pick lowest threshold on K1 held-out with recall >= 0.98 and TNR >= 0.60
    ok = [r for r in k1_rows if r[1] >= 0.98 and r[3] >= 0.60]
    chosen = max(ok, key=lambda r: r[3]) if ok else None
    if chosen:
        thr = chosen[0]
        print(f"\n  CHOSEN threshold {thr:.2f}: K1 held-out recall {chosen[1]:.3f}, "
              f"cut {chosen[3]:.1%} of non-conflicts (recall>=0.98 AND cut>=60%)")
    else:
        # fall back to the lowest threshold that still hits recall >= 0.98
        rec_ok = [r for r in k1_rows if r[1] >= 0.98]
        chosen = max(rec_ok, key=lambda r: r[3]) if rec_ok else k1_rows[0]
        thr = chosen[0]
        print(f"\n  No threshold meets recall>=0.98 AND cut>=60% on K1 held-out. "
              f"Best recall>=0.98 point: thr {thr:.2f}, cut {chosen[3]:.1%} of non-conflicts.")

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    joblib.dump({"embed_model": EMBED, "kind": "lr", "clf": clf, "threshold": thr,
                 "features": "a,b,|a-b|,a*b"}, os.path.expanduser(args.out))
    print(f"  saved -> {args.out} (threshold {thr:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
