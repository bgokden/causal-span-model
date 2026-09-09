"""Merge generated chunks + CNC into train/dev/test CSVs the pointer trainer reads.

Split is by seed pair (``pair_id``): no cause/effect phrase crosses splits. Stratified
by language and domain. CNC keeps its own official train/dev. Output:

    <out>/train.csv       synthetic train rows (text_w_pairs / num_rs)  -> --extra
    <out>/negatives.csv   synthetic non-causal rows                     -> --negatives
    <out>/dev.csv, test.csv   synthetic dev/test (test: keep for the human check)
    <out>/stats.json

    python scripts/prepare_splits.py --chunks data/synth --cnc data/cnc --out data/prepared
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# Language purity (lab C3/C4): drop non-English CAUSAL rows whose text carries the relation with an
# English connective. A load-time filter so re-running never reintroduces the code-mix defect even from
# old chunks; the generator also rejects these now (scripts/gen_causal_data.py).
EN_RELATION_WORDS = [
    "was due to", "due to", "because of", "partly because", r"\bbecause\b", "may have caused", "which caused",
    r"\bcaused\b", r"\bcausing\b", "led to", "leads to", "resulted in", "resulted from", "result of",
    "contributed to", r"\bcontributed\b", "in order to", "so that", r"\bprevented\b", r"\bprevents\b",
    r"\btriggered\b", r"\btriggers\b", r"\ballowed\b", r"\benabled\b", "as a result", r"\btherefore\b",
    r"\bthus\b", "consequently", "owing to", "thanks to", "according to",
    # NOT in this list, deliberately: bare "so" and "with". As a permanent load-time filter they
    # delete legitimate rows -- German "so dass" and the same bare token inside other languages --
    # while catching almost nothing: of the 931 rows the C4 audit quarantined, only 3 were caught by
    # these two alone, and 13 German/Dutch rows matched bare "so". Precision over one percent of recall.
]
EN_RELATION_RX = re.compile("|".join(EN_RELATION_WORDS), re.I)


def load_chunks(root: str) -> tuple[list[dict], int]:
    rows, dropped = [], 0
    for f in glob.glob(f"{root}/**/synth.jsonl", recursive=True):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("lang", "en") != "en" and r.get("kind") == "causal" \
                    and EN_RELATION_RX.search(r.get("text", "") or ""):
                dropped += 1
                continue
            r["_chunk"] = f
            rows.append(r)
    return rows, dropped


def to_text_w_pairs(row: dict) -> str:
    s = row["text"]
    spans = [(row["cause_span"], "ARG0"), (row["effect_span"], "ARG1")]
    if row.get("signal_span"):
        spans.append((row["signal_span"], "SIG0"))
    out, last = [], 0
    for (a, b), tag in sorted(spans, key=lambda t: t[0][0]):
        out.append(s[last:a]); out.append(f"<{tag}>{s[a:b]}</{tag}>"); last = b
    out.append(s[last:])
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/synth")
    ap.add_argument("--cnc", default="data/cnc")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--dev", type=float, default=0.08)
    ap.add_argument("--test", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--langs", default="", help="comma list, e.g. en or en,de; empty = all languages")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows, codemix_dropped = load_chunks(args.chunks)
    # global dedupe (chunks from different providers may overlap)
    langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    seen, uniq = set(), []
    for r in rows:
        if langs and r.get("lang", "en") not in langs:
            continue
        key = hashlib.sha1(" ".join(r["text"].lower().split()).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
    # group by pair id (negatives without a pair id get their own bucket)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in uniq:
        # same-phrase negatives share their pair's id (so they follow it into one split);
        # plain facts have no pair and must not collapse into one group per domain
        if r["kind"] == "causal" or r.get("a"):
            pid = r.get("pair_id") or hashlib.sha1(r["text"].encode()).hexdigest()[:10]
        else:
            pid = "plain-" + hashlib.sha1(r["text"].encode()).hexdigest()[:10]
        groups[pid].append(r)
    # stratified assignment by (lang, domain)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for pid, rs in groups.items():
        strata[(rs[0].get("lang", "en"), rs[0].get("domain", "?"))].append(pid)
    split_of: dict[str, str] = {}
    for key, pids in strata.items():
        rng.shuffle(pids)
        n = len(pids); n_test = int(n * args.test); n_dev = int(n * args.dev)
        for i, pid in enumerate(pids):
            split_of[pid] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    writers = {}
    files = {}
    for name in ("train", "dev", "test", "negatives"):
        files[name] = open(out / f"{name}.csv", "w", newline="", encoding="utf-8")
        writers[name] = csv.writer(files[name]); writers[name].writerow(["index", "text", "text_w_pairs", "num_rs", "lang", "domain", "register", "style", "pair_id"])
    counts = Counter(); i = 0
    for pid, rs in groups.items():
        split = split_of[pid]
        for r in rs:
            causal = r["kind"] == "causal"
            twp = to_text_w_pairs(r) if causal else r["text"]
            target = split if causal else ("negatives" if split == "train" else split)
            writers[target].writerow([i, r["text"], twp, r.get("num_rs", 1) if causal else 0, r.get("lang", "en"),
                                      r.get("domain", "?"), r.get("register", ""), r.get("style", ""), pid])
            counts[(target, "causal" if causal else "negative")] += 1; i += 1
    for f in files.values():
        f.close()
    stats = {"raw_rows": len(rows), "codemix_dropped": codemix_dropped,
             "unique_rows": len(uniq), "pairs": len(groups),
             "counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
             "langs": dict(Counter(r.get("lang", "en") for r in uniq)),
             "domains": dict(Counter(r.get("domain") for r in uniq)),
             "registers": dict(Counter(r.get("register") for r in uniq if r["kind"] == "causal")),
             "cnc_train": str(Path(args.cnc) / "train_subtask2_grouped.csv"),
             "cnc_dev": str(Path(args.cnc) / "dev_subtask2_grouped.csv")}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
