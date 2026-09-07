"""Prepare the two derived datasets the multi-task LLM trainer needs.

Everything else (CNC causal, prepared multilingual causal, the negative sets, the
1012 conflict pairs) is read straight from CSV/JSONL by ``train_llm_multitask.py``.
Two inputs need an external model, so they are generated once and cached here:

  --entity-silver         GLiNER (reasongraph's entity extractor) over the synthetic
                          sentences -> data/llm/entity_silver.jsonl. Run in the
                          reasongraph venv (needs gliner + reasongraph).

  --conflict-paraphrases  N local-LLM paraphrases per hand contradiction pair
                          (rg/tests/data/contradictions.jsonl), labels kept, the 40
                          originals held out for eval -> data/llm/conflict_paraphrases.jsonl.
                          Needs a local OpenAI-compatible endpoint (Ollama).

Usage:
    # reasongraph venv:
    uv run --project ~/repos/reasongraph python scripts/prep_llm_data.py --entity-silver
    # csm venv, Ollama up:
    LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=gpt-oss:20b \
        uv run python scripts/prep_llm_data.py --conflict-paraphrases --per-pair 5
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRADICTIONS = os.path.expanduser("~/repos/reasongraph/tests/data/contradictions.jsonl")
OUT_DIR = os.path.join(REPO, "data", "llm")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def entity_silver(sources, out_path, limit=None):
    """Silver entity labels: rg's GlinerExtractor over unique synthetic sentences."""
    from reasongraph._extraction import GlinerExtractor

    texts = []
    seen = set()
    for src in sources:
        df = pd.read_csv(src, encoding="utf-8-sig")
        for t in df["text"]:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
    if limit:
        texts = texts[:limit]
    print(f"entity-silver: {len(texts)} unique sentences")

    extractor = GlinerExtractor()  # gliner_small-v2.5, person/org/location/event
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, text in enumerate(texts):
            ents = extractor(text)
            fh.write(json.dumps({"text": text, "entities": ents}, ensure_ascii=False) + "\n")
            n += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(texts)}", flush=True)
    print(f"wrote {n} -> {out_path}")


def conflict_paraphrases(out_path, per_pair=5):
    """Paraphrase the 40 hand pairs; keep labels; hold out the 40 originals."""
    from gen_causal_data import LLM

    llm = LLM(os.environ["LLM_BASE_URL"], os.environ.get("LLM_API_KEY", "ollama"),
              os.environ["LLM_MODEL"])
    pairs = _read_jsonl(CONTRADICTIONS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for idx, p in enumerate(pairs):
        rel = ("CONTRADICT each other (one makes the other false)" if p["label"]
               else "are COMPATIBLE (both can be true at once)")
        prompt = (
            f"Two short facts that {rel}:\n"
            f"A: {p['old']}\nB: {p['new']}\n\n"
            f"Write {per_pair} new paraphrase pairs that keep the SAME relationship "
            f"(still {'contradicting' if p['label'] else 'compatible'}), rewording both "
            f"facts and varying names/entities/wording. Return a JSON object only: "
            f'{{"pairs": [{{"old": "...", "new": "..."}}, ...]}}')
        try:
            obj = llm.json(prompt, max_tokens=1200)
            items = obj.get("pairs", obj) if isinstance(obj, dict) else obj
            got = 0
            for pr in list(items)[:per_pair]:
                old, new = str(pr.get("old", "")).strip(), str(pr.get("new", "")).strip()
                if old and new:
                    rows.append({"old": old, "new": new, "label": bool(p["label"]),
                                 "kind": p.get("kind", ""), "source": "paraphrase"})
                    got += 1
            print(f"  pair {idx} ({'C' if p['label'] else 'x'}) -> {got}", flush=True)
        except Exception as e:
            print(f"  pair {idx} error: {e}", flush=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} paraphrases -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity-silver", action="store_true")
    ap.add_argument("--conflict-paraphrases", action="store_true")
    ap.add_argument("--per-pair", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    if args.entity_silver:
        entity_silver([os.path.join(REPO, "data/prepared/train.csv"),
                       os.path.join(REPO, "data/prepared/negatives.csv")],
                      os.path.join(OUT_DIR, "entity_silver.jsonl"), limit=args.limit)
    if args.conflict_paraphrases:
        conflict_paraphrases(os.path.join(OUT_DIR, "conflict_paraphrases.jsonl"),
                             per_pair=args.per_pair)
    if not (args.entity_silver or args.conflict_paraphrases):
        ap.error("pick --entity-silver and/or --conflict-paraphrases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
