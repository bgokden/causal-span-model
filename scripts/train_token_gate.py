"""G3: token-aware end-to-end causal gate.

The frozen-embedding gate (G1) encodes topic, not the relation, so it misses verb-causal
sentences ("The GPU cluster increased training throughput", "Redis runs on the same node,
so the rebuild saturates the CPU"): a paraphrase embedding puts them next to their
non-causal twin. This fine-tunes the `paraphrase-multilingual-MiniLM-L12-v2` backbone with
a sequence-classification head (3 epochs, lr 3e-5) so the model attends to causal tokens,
and adds an LLM-generated verb-causal positive slice to the G1 training data.

Data (run from the csm repo root so the relative paths resolve, same as G1):
  CNC subtask-1 (0/1) + synthetic positives + synthetic/prose negatives + verb-causal
  rewrites. Held out: CNC-1 dev + synthetic dev.

Verb-causal generation uses the LOCAL model only (never Groq):
  LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=gpt-oss:20b

Run:
  cd ~/repos/causal-span-model
  LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=gpt-oss:20b \
    python scripts/train_token_gate.py \
      --out ~/repos/primaxiom-lab/causal/models/gate_token_minilm \
      --verb-cache ~/repos/primaxiom-lab/causal/data/llm/verb_causal.jsonl \
      --reviewed ~/repos/primaxiom-lab/eval/causal_cases_reviewed.jsonl \
      --plain ~/repos/primaxiom-lab/eval/plain_facts.txt \
      --g1-mlp ~/repos/primaxiom-lab/causal/models/gate_paraphrase-multilingual-MiniLM-L12-v2_mlp.joblib
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LANG_NAMES = {"en": "English", "de": "German", "nl": "Dutch", "es": "Spanish",
              "fr": "French", "tr": "Turkish"}


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ----------------------------------------------------------------------------- data

def build_g1_rows():
    """Exactly the G1 gate data (train, dev), each item (text, label, lang, domain)."""
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
        dev.append((str(r["text"]), 1 if int(r["num_rs"]) > 0 else 0,
                    r.get("lang", "en"), r.get("domain", "?")))
    return train, dev


def sample_causal_source(n, seed=42):
    """n causal synthetic rows (num_rs>0), stratified across languages present."""
    rows = [r for r in _rows("data/prepared/train.csv") if int(r.get("num_rs", 0)) > 0]
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r.get("lang", "en"), []).append(r)
    rng = random.Random(seed)
    out = []
    langs = sorted(by_lang)
    per = max(1, n // len(langs))
    for lang in langs:
        pool = by_lang[lang]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out[:n]


VERB_PROMPT = """Rewrite each numbered sentence as ONE natural sentence in {lang} that states
the SAME cause and effect but using a CAUSAL VERB or clause instead of an explicit connective.
Use forms like: X increased/raised/reduced/cut/drove/triggered/led to/resulted in/caused Y,
or "after X, Y", or "X, which caused Y". Do NOT use the words "because", "since", "as a result"
as the link. Keep it factual, self-contained, no pronouns referring outside the sentence.
Return a JSON array of {n} strings, in the same order, nothing else.

Sentences:
{items}"""


def gen_verb_causal(n, cache_path, batch=12):
    """Generate/cache n verb-causal positive rewrites via the local LLM. Idempotent:
    reuses an existing cache and only tops it up to n."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from gen_causal_data import LLM

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    have = _jsonl(cache_path) if os.path.exists(cache_path) else []
    if len(have) >= n:
        print(f"verb-causal cache has {len(have)} >= {n}; reusing", flush=True)
        return have[:n]

    llm = LLM(os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
              os.environ.get("LLM_API_KEY", "ollama"),
              os.environ.get("LLM_MODEL", "gpt-oss:20b"))
    src = sample_causal_source(n)
    src = src[len(have):]  # only generate the shortfall
    print(f"generating {len(src)} verb-causal rewrites (batch {batch}) -> {cache_path}", flush=True)
    fh = open(cache_path, "a", encoding="utf-8")
    for i in range(0, len(src), batch):
        chunk = src[i:i + batch]
        # group by language so one prompt is one language
        by_lang = {}
        for r in chunk:
            by_lang.setdefault(r.get("lang", "en"), []).append(r)
        for lang, rs in by_lang.items():
            items = "\n".join(f"{j + 1}. {r['text']}" for j, r in enumerate(rs))
            try:
                out = llm.json(VERB_PROMPT.format(lang=LANG_NAMES.get(lang, "English"),
                                                  n=len(rs), items=items),
                               max_tokens=1600)
            except Exception as exc:
                print(f"  batch {i} ({lang}) failed: {exc}", flush=True)
                continue
            if not isinstance(out, list):
                continue
            for r, rewrite in zip(rs, out):
                text = str(rewrite).strip()
                if not text or len(text) < 8:
                    continue
                rec = {"text": text, "lang": lang, "label": 1, "src": r["text"]}
                have.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  ...{len(have)}/{n}", flush=True)
    fh.close()
    return have[:n]


CONTRAST_PROMPT = """Generate {n} contrastive sentence pairs in {lang} for training a
causal-relation detector. Each pair is about the SAME subject:
- "causal": a sentence stating a cause->effect using a VARIED causal verb or clause
  (increased, reduced, cut, raised, lowered, caused, led to, triggered, produced, created,
  released, warmed, cooled, damaged, weakened, enabled, prevented, made X do Y, drove,
  slowed, accelerated, "after X, Y", "X, which caused Y"). Do NOT rely on "because"/"since".
- "plain": a sentence about the SAME subject with NO causal relation -- a location,
  attribute, definition, quantity, schedule, or plain action with no stated effect.
Cover varied domains: science, physics, health, economics, technology, logistics, daily life.
Keep sentences short, factual, self-contained (no pronouns referring outside the sentence).
Return a JSON array of {n} objects {{"causal": "...", "plain": "..."}}, nothing else."""


def gen_contrastive_causal(n, cache_path, batch=10):
    """Generate/cache n contrastive (causal, plain-twin) pairs via the local LLM.
    Teaches relation-vs-topic separation directly. Idempotent top-up."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from gen_causal_data import LLM

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    have = _jsonl(cache_path) if os.path.exists(cache_path) else []
    if len(have) >= n:
        print(f"contrastive cache has {len(have)} >= {n}; reusing", flush=True)
        return have[:n]

    llm = LLM(os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
              os.environ.get("LLM_API_KEY", "ollama"),
              os.environ.get("LLM_MODEL", "gpt-oss:20b"))
    langs = ["en", "en", "en", "de", "es", "fr", "nl", "tr"]  # English-weighted
    fh = open(cache_path, "a", encoding="utf-8")
    li = 0
    while len(have) < n:
        lang = langs[li % len(langs)]
        li += 1
        try:
            out = llm.json(CONTRAST_PROMPT.format(n=batch, lang=LANG_NAMES.get(lang, "English")),
                           max_tokens=1600)
        except Exception as exc:
            print(f"  contrastive batch ({lang}) failed: {exc}", flush=True)
            continue
        if not isinstance(out, list):
            continue
        for pair in out:
            if not isinstance(pair, dict):
                continue
            c = str(pair.get("causal", "")).strip()
            p = str(pair.get("plain", "")).strip()
            if len(c) < 8 or len(p) < 8:
                continue
            rec = {"causal": c, "plain": p, "lang": lang}
            have.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"  ...{len(have)}/{n} contrastive pairs", flush=True)
    fh.close()
    return have[:n]


# ----------------------------------------------------------------------------- model

def score_texts(model, tok, texts, device, batch=128, max_len=96):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, 1]
            out.extend(p.cpu().tolist())
    return np.array(out)


def f1_at(y, scores, cutoff):
    from sklearn.metrics import f1_score, accuracy_score
    pred = (scores >= cutoff).astype(int)
    return accuracy_score(y, pred), f1_score(y, pred, zero_division=0)


# ----------------------------------------------------------------------------- eval sets

def load_reviewed_hop_facts(path):
    facts = []
    for c in _jsonl(path):
        for f in c.get("gold_chain", []):
            if str(f).strip():
                facts.append(str(f).strip())
    return facts


def load_plain_facts(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def g1_mlp_scores(joblib_path, texts):
    import joblib
    from sentence_transformers import SentenceTransformer
    bundle = joblib.load(joblib_path)
    st = SentenceTransformer(bundle["embed_model"], device="cuda")
    emb = st.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    return bundle["clf"].predict_proba(emb)[:, 1]


def sweep(name, hop_scores, plain_scores, cutoffs):
    print(f"  [{name}] cutoff sweep (hop kept / plain rejected):")
    rows = []
    for c in cutoffs:
        hop_kept = float((hop_scores >= c).mean())
        plain_rej = float((plain_scores < c).mean())
        passes = hop_kept >= 0.95 and plain_rej >= 0.95
        rows.append((c, hop_kept, plain_rej, passes))
        print(f"    {c:.2f}: hop {hop_kept:.3f} / plain_rej {plain_rej:.3f}"
              f"{'  <== PASS' if passes else ''}")
    return rows


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verb-cache", required=True)
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--g1-mlp", required=True)
    ap.add_argument("--n-verb", type=int, default=800)
    ap.add_argument("--n-contrast", type=int, default=0,
                    help="contrastive (causal, plain-twin) pairs to add; 0 = off")
    ap.add_argument("--contrast-cache",
                    default="~/repos/primaxiom-lab/causal/data/llm/contrastive_causal.jsonl")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-train", action="store_true", help="load --out and only eval")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    verb = gen_verb_causal(args.n_verb, os.path.expanduser(args.verb_cache))
    train, dev = build_g1_rows()
    train = train + [(r["text"], 1, r.get("lang", "en"), "verb-causal") for r in verb]
    contrast = []
    if args.n_contrast > 0:
        contrast = gen_contrastive_causal(args.n_contrast, os.path.expanduser(args.contrast_cache))
        for r in contrast:
            train.append((r["causal"], 1, r.get("lang", "en"), "contrast"))
            train.append((r["plain"], 0, r.get("lang", "en"), "contrast"))
    random.Random(args.seed).shuffle(train)
    ytr = np.array([t[1] for t in train])
    ydv = np.array([t[1] for t in dev])
    print(f"train {len(train)} (pos {int(ytr.sum())} / neg {int((1 - ytr).sum())}) "
          f"| dev {len(dev)} (pos {int(ydv.sum())}) | verb-causal {len(verb)} "
          f"| contrastive pairs {len(contrast)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    out_dir = os.path.expanduser(args.out)

    if not args.skip_train:
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

        def tokenize(examples):
            return tok(examples["text"], truncation=True, max_length=args.max_len)

        ds_tr = Dataset.from_dict({"text": [t[0] for t in train], "label": ytr.tolist()}).map(
            tokenize, batched=True, remove_columns=["text"])
        ds_dv = Dataset.from_dict({"text": [t[0] for t in dev], "label": ydv.tolist()}).map(
            tokenize, batched=True, remove_columns=["text"])

        from transformers import DataCollatorWithPadding
        collator = DataCollatorWithPadding(tok)

        targs = TrainingArguments(
            output_dir=os.path.join(out_dir, "_ckpt"),
            num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=256,
            bf16=torch.cuda.is_available(), logging_steps=50, report_to=[],
            save_strategy="no", seed=args.seed)
        trainer = Trainer(model=model, args=targs, train_dataset=ds_tr,
                          data_collator=collator, tokenizer=tok)
        trainer.train()
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        print(f"saved token gate -> {out_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)

    # (a) held-out dev F1 at argmax (0.5)
    dev_scores = score_texts(model, tok, [t[0] for t in dev], device,
                             batch=256, max_len=args.max_len)
    cnc_idx = [i for i, t in enumerate(dev) if t[3] == "news"]
    syn_idx = [i for i, t in enumerate(dev) if t[3] != "news"]
    acc, f1 = f1_at(ydv, dev_scores, 0.5)
    ca, cf = f1_at(ydv[cnc_idx], dev_scores[cnc_idx], 0.5)
    sa, sf = f1_at(ydv[syn_idx], dev_scores[syn_idx], 0.5)
    print(f"\n(a) dev @0.5: overall acc {acc:.3f} f1 {f1:.3f} | "
          f"CNC-1 f1 {cf:.3f} (n={len(cnc_idx)}) | synth f1 {sf:.3f} (n={len(syn_idx)})")

    # (b)/(c) reviewed hop facts + plain facts
    hop = load_reviewed_hop_facts(os.path.expanduser(args.reviewed))
    plain = load_plain_facts(os.path.expanduser(args.plain))
    hop_s = score_texts(model, tok, hop, device, batch=256, max_len=args.max_len)
    plain_s = score_texts(model, tok, plain, device, batch=256, max_len=args.max_len)
    print(f"\n(b)/(c) reviewed hop facts n={len(hop)} | plain facts n={len(plain)}")
    cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    token_rows = sweep("token-gate", hop_s, plain_s, cutoffs)

    # G1 MLP comparison on the same three sets
    print("\n[G1 MLP baseline] same sets")
    g1_dev = g1_mlp_scores(os.path.expanduser(args.g1_mlp), [t[0] for t in dev])
    ga, gf = f1_at(ydv, g1_dev, 0.5)
    print(f"  (a) dev @0.5: acc {ga:.3f} f1 {gf:.3f}")
    g1_hop = g1_mlp_scores(os.path.expanduser(args.g1_mlp), hop)
    g1_plain = g1_mlp_scores(os.path.expanduser(args.g1_mlp), plain)
    g1_rows = sweep("G1-MLP", g1_hop, g1_plain, cutoffs)

    # (d) CPU latency
    cpu_model = AutoModelForSequenceClassification.from_pretrained(out_dir).to("cpu")
    torch.set_num_threads(4)
    sample = (hop + plain)[:60]
    _ = score_texts(cpu_model, tok, sample[:4], "cpu", batch=1, max_len=args.max_len)  # warmup
    lat = []
    for s in sample:
        t0 = time.perf_counter()
        score_texts(cpu_model, tok, [s], "cpu", batch=1, max_len=args.max_len)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    print(f"\n(d) CPU ms/sentence (4 threads): median {lat[len(lat)//2]:.1f} "
          f"p90 {lat[int(0.9*(len(lat)-1))]:.1f} (n={len(lat)})")

    # pick a recommended cutoff for the token gate
    passing = [r for r in token_rows if r[3]]
    if passing:
        rec = max(passing, key=lambda r: (r[1] + r[2]))
        print(f"\nRECOMMEND token-gate cutoff {rec[0]:.2f}: hop kept {rec[1]:.3f}, "
              f"plain rejected {rec[2]:.3f} (PASS: hop>=0.95 AND plain_rej>=0.95)")
    else:
        best = max(token_rows, key=lambda r: min(r[1], r[2]))
        print(f"\nNo cutoff clears hop>=0.95 AND plain_rej>=0.95. Best balance at "
              f"{best[0]:.2f}: hop {best[1]:.3f}, plain_rej {best[2]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
