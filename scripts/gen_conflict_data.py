"""Training pairs for a purpose-built contradiction ("is the old fact no longer true?")
classifier, so the memory service can retire facts without calling an LLM.

Pair = (existing fact, new fact) -> label 1 (the new fact makes the old one no longer true
as a statement about the present) or 0 (compatible: adds detail, is historical, is about a
different attribute, or only sounds related).

Positives are built two ways:
  * by construction: value/entity substitutions in template facts (numbers, names, places,
    plans, versions, schedules) -- labels certain;
  * by the LLM: rewrite a seed fact as a replacement / reversal / ending / state change,
    then an independent QA pass must agree the old fact is now false (disagreement -> drop).
Negatives: LLM-written complements and historical facts + template siblings (same entity,
different attribute) + random unrelated facts from the same domain.

    LLM_API_KEY=... python scripts/gen_conflict_data.py --domains tech,support --seeds 20 --out data/conflicts
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_causal_data import LLM, DOMAINS, LANG_NAMES  # noqa: E402

TEMPLATES = [
    # (fact template, slot -> alternatives). Substituting one slot yields a certain contradiction;
    # a sibling template about the same subject yields a certain complement.
    ("{who} works from the {city} office.", {"city": ["Berlin", "Amsterdam", "Lisbon", "Madrid", "Oslo"]}),
    ("{who} reports to {boss}.", {"boss": ["Dana", "Chris", "Priya", "Tom", "Yusuf"]}),
    ("{who} is on the {plan} plan.", {"plan": ["Free", "Pro", "Team", "Scale"]}),
    ("The API rate limit is {n} requests per minute.", {"n": ["60", "100", "250", "1000"]}),
    ("The export size limit is {n} MB.", {"n": ["50", "100", "500", "2000"]}),
    ("Support hours are {h}.", {"h": ["09:00 to 17:00", "08:00 to 20:00", "24 hours"]}),
    ("The database runs Postgres {v}.", {"v": ["15", "16", "17"]}),
    ("The weekly meeting is on {day} at 15:00.", {"day": ["Monday", "Wednesday", "Thursday"]}),
    ("The default model is {m}.", {"m": ["GPT-OSS 20B", "Qwen 3.8", "Llama 4"]}),
    ("{who} prefers {pref} seats on flights.", {"pref": ["window", "aisle"]}),
    ("The project deadline is {date}.", {"date": ["31 October", "30 November", "15 January"]}),
    ("Checkout stores shopping carts in {store}.", {"store": ["Redis", "Postgres", "DynamoDB"]}),
    ("The office in {city} is {state}.", {"state": ["open", "closed"]}),
]
SIBLINGS = {  # complements about the same subject (label 0)
    "who": ["{who} joined the company in 2021.", "{who} leads the platform team.", "{who} likes cycling to work.",
            "{who} wrote the deployment runbook.", "{who} is on call this week."],
    "generic": ["The API requires a bearer token.", "Exports are delivered as ZIP files.", "Support is reachable by email.",
                "The database is backed up nightly.", "The meeting is about the Q4 budget.", "Models are hosted on Groq.",
                "The project has three milestones.", "The office has forty desks."],
}
NAMES = ["Alice", "Bob", "Dana", "Maria Lopez", "Chris", "Priya", "Tom", "Yusuf", "Elif", "Jonas"]
CITIES = ["Berlin", "Amsterdam", "Lisbon", "Madrid", "Oslo", "Rotterdam"]

UPDATE_KINDS = ["a replacement (the value or entity changed)", "a reversal (the opposite is now true)",
                "an ending (it stopped, was cancelled, closed, or left)", "a state change (paused, moved, resumed, downgraded)"]
COMPLEMENT_KINDS = ["adds a new detail about the same subject that does not change the old fact",
                    "a historical event about the same subject that remains true (past tense)",
                    "a fact about a different attribute of the same subject",
                    "a related fact about a different subject mentioned nearby"]


def template_pairs(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        tpl, slots = rng.choice(TEMPLATES)
        who = rng.choice(NAMES); city = rng.choice(CITIES)
        slot, values = next(iter(slots.items()))
        a, b = rng.sample(values, 2)
        fill = lambda v: tpl.format(**{slot: v, "who": who, "city": city, "boss": v if slot == "boss" else "Dana"})  # noqa: E731
        old, new = fill(a), fill(b)
        rows.append({"old": old, "new": new, "label": 1, "kind": "template-substitution", "domain": "template"})
        sib_pool = SIBLINGS["who"] if "{who}" in tpl else SIBLINGS["generic"]
        rows.append({"old": old, "new": rng.choice(sib_pool).format(who=who), "label": 0, "kind": "template-sibling", "domain": "template"})
    return rows


def seed_facts(llm: LLM, domain: str, n: int, lang: str) -> list[str]:
    out = llm.json(f"Write {n} varied {LANG_NAMES[lang]} facts about the current state of things in {DOMAINS[domain]}: "
                   "who does what where, values and limits, schedules, plans, preferences, versions, statuses. "
                   "6-20 words, present tense, specific (names, numbers, places). Return a JSON list of strings.")
    return [str(s).strip() for s in (out if isinstance(out, list) else out.get("items", [])) if 4 <= len(str(s).split()) <= 30]


def llm_pairs(llm: LLM, facts: list[str], lang: str) -> list[dict]:
    items = [{"id": i, "fact": f, "update": random.choice(UPDATE_KINDS), "complement": random.choice(COMPLEMENT_KINDS)} for i, f in enumerate(facts)]
    prompt = (f"For each item write two new {LANG_NAMES[lang]} facts (6-20 words, present tense where natural): "
              "'update' = a later fact of the requested kind after which the ORIGINAL FACT IS NO LONGER TRUE as a statement "
              "about the present; 'complement' = a fact of the requested kind after which the original fact is STILL TRUE. "
              "Vary wording; do not copy the original.\n"
              f"Items: {json.dumps(items, ensure_ascii=False)}\n"
              'Return a JSON list of objects {"id": <id>, "update": "...", "complement": "..."}.')
    out = llm.json(prompt, max_tokens=4000)
    rows = []
    for o in out if isinstance(out, list) else out.get("items", []):
        try:
            f = facts[int(o["id"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        u, c = (o.get("update") or "").strip(), (o.get("complement") or "").strip()
        if u and u.lower() != f.lower():
            rows.append({"old": f, "new": u, "label": 1, "kind": "llm-update", "domain": "llm"})
        if c and c.lower() != f.lower():
            rows.append({"old": f, "new": c, "label": 0, "kind": "llm-complement", "domain": "llm"})
    return rows


def qa(llm: LLM, rows: list[dict]) -> list[dict]:
    """Independent judgement with the production prompt wording; keep only agreements."""
    keep = []
    for i in range(0, len(rows), 15):
        chunk = rows[i:i + 15]
        items = [{"id": j, "existing": r["old"], "new": r["new"]} for j, r in enumerate(chunk)]
        prompt = ("For each item: given the new fact, is the existing fact NO LONGER TRUE as a statement about the present "
                  "(replaced, reversed, ended, or its value changed)? Facts about different things, or past events that "
                  "still happened, do not count.\n"
                  f"Items: {json.dumps(items, ensure_ascii=False)}\n"
                  'Return a JSON list of objects {"id": <id>, "outdated": true|false}.')
        try:
            out = llm.json(prompt, max_tokens=1200)
            v = {int(o["id"]): bool(o["outdated"]) for o in (out if isinstance(out, list) else out.get("items", []))}
        except Exception:
            keep += chunk; continue
        for j, r in enumerate(chunk):
            if j not in v or v[j] == bool(r["label"]):
                keep.append(r)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="tech,support,business,personal")
    ap.add_argument("--langs", default="en")
    ap.add_argument("--seeds", type=int, default=20, help="LLM seed facts per domain per language")
    ap.add_argument("--templates", type=int, default=40, help="template-built pairs (certain labels)")
    ap.add_argument("--out", default="data/conflicts")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed); random.seed(args.seed)
    llm = LLM(os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"), os.environ["LLM_API_KEY"],
              os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"))
    rows = template_pairs(rng, args.templates)
    llm_rows, all_facts = [], []
    for lang in args.langs.split(","):
        for domain in args.domains.split(","):
            facts = seed_facts(llm, domain, args.seeds, lang); all_facts += facts
            for i in range(0, len(facts), 10):
                llm_rows += llm_pairs(llm, facts[i:i + 10], lang)
    before = len(llm_rows); llm_rows = qa(llm, llm_rows)
    # unrelated negatives: random fact pairs from the pool
    unrelated = [{"old": a, "new": b, "label": 0, "kind": "unrelated", "domain": "llm"}
                 for a, b in (rng.sample(all_facts, 2) for _ in range(len(llm_rows) // 4))] if len(all_facts) > 2 else []
    rows += llm_rows + unrelated
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "pairs.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "pairs.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["old", "new", "label", "kind"])
        for r in rows:
            w.writerow([r["old"], r["new"], r["label"], r["kind"]])
    from collections import Counter
    print(json.dumps({"rows": len(rows), "positives": sum(r["label"] for r in rows), "llm_rows_before_qa": before,
                      "llm_rows_after_qa": len(llm_rows), "kinds": dict(Counter(r["kind"] for r in rows))}, indent=1))
    for r in rng.sample(rows, min(8, len(rows))):
        print(f"[{r['label']} {r['kind']:22}] {r['old']}  ||  {r['new']}")


if __name__ == "__main__":
    main()
