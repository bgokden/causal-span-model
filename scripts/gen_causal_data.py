"""Synthetic causal-span data whose labels are correct by construction.

The LLM never draws span boundaries. It (1) proposes cause/effect *event phrases*
per domain and (2) writes natural sentences that must contain those phrases
verbatim; a sentence is accepted only if both phrases occur exactly once, do not
overlap, and the sentence passes length/dedup checks. Signal spans come from a
connective lexicon matched between/around the phrases. Output is the CNC
``text_w_pairs`` format both trainers consume, plus JSONL.

    LLM_API_KEY=... python scripts/gen_causal_data.py --domains tech,support --pairs 20 \
        --per-pair 2 --negatives 0.3 --langs en --out data/synth

Env: LLM_BASE_URL (default Groq), LLM_API_KEY, LLM_MODEL (default openai/gpt-oss-120b).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

DOMAINS = {
    "tech": "software engineering, infrastructure, incidents, deployments, databases, APIs",
    "support": "customer support, SaaS accounts, billing, feature limits, churn, tickets",
    "business": "company operations, sales, suppliers, contracts, hiring, budgets",
    "personal": "everyday life, travel, appointments, preferences, household, health habits",
    "news": "economy, environment, politics, industry, public infrastructure",
    "health": "medicine, symptoms, treatments, fitness, nutrition, hospitals",
    "finance": "personal finance, markets, loans, payments, taxes, pricing",
    "logistics": "shipping, warehouses, delays, suppliers, transport, inventory",
    "science": "experiments, lab equipment, measurements, materials, climate",
    "research": "academic research, papers, grants, datasets, peer review",
}

# Text registers an agent actually stores: not only tidy prose.
REGISTERS = {
    "prose":   "a plain written sentence, like a report or an article",
    "chat":    "an informal chat or Slack message: contractions, lowercase start allowed, maybe an emoji or 'tbh', no hashtags",
    "note":    "a terse incident or meeting note: fragments, abbreviations, a timestamp or ticket id somewhere",
    "firstperson": "a first-person statement from someone's own day ('I ...', 'we ...', 'my ...')",
    "qa":      "a question followed by its answer in the same line, e.g. 'Why did X happen? Because Y.'",
    "email":   "one sentence from a polite work email",
    "log":     "a log line or alert text: prefixed with a level or timestamp, terse",
}
# Extra causal phrasings that a plain 'because' corpus under-represents (all causal per CNC):
# Styles whose wording only works when the phrases are noun phrases ("prevented the
# outage", "in order to reduce latency"); clause phrases would come out ungrammatical.
NOMINAL_ONLY_STYLES = {4, 5}   # indices into CAUSAL_STYLES: prevention/enablement, purpose
CAUSAL_STYLES = [
    "explicit connective, cause first",
    "explicit connective, effect first (e.g. 'X resulted from Y', 'X was due to Y', 'X, because Y')",
    "implicit: no connective at all, causality understood from the two clauses",
    "hedged or partial causation ('partly because', 'probably due to', 'contributed to', 'may have caused')",
    "prevention or enablement ('prevented', 'stopped', 'allowed', 'made it possible')",
    "purpose ('in order to', 'so that', 'to be able to')",
    "with a temporal detail as context but a causal link as the connection (e.g. 'On Monday ... because ...')",
    "with numbers or a named entity added as context (a percentage, a date, a product or person name)",
]

# Connective lexicon with the direction it implies. Matched as a signal span when
# present between/around the argument phrases. Purpose ("in order to") follows CNC:
# labelled causal.
SIGNALS = {
    "en": ["because of", "because", "due to", "as a result of", "as a result", "led to", "leads to",
           "caused", "causes", "resulted in", "results in", "so that", "so", "therefore", "thanks to",
           "owing to", "triggered", "forced", "since", "hence", "consequently", "in order to", "after"],
    "de": ["weil", "wegen", "aufgrund", "führte zum", "führte zur", "führten zum", "führten zur", "führte zu",
           "führt zu", "verursachte", "verursachten", "resultierte aus", "resultierten aus", "kam es zur",
           "kam es zum", "durch", "deshalb", "daher", "sodass", "infolge", "bedingt durch", "löste", "lösten"],
    "nl": ["omdat", "door", "vanwege", "leidde tot", "leidt tot", "leidden tot", "veroorzaakte", "veroorzaakten",
           "daardoor", "dus", "waardoor", "als gevolg van", "zorgde voor", "zorgden voor"],
    "es": ["porque", "debido a", "a causa de", "provocó", "causó", "llevó a", "por lo que", "así que", "gracias a"],
    "fr": ["parce que", "à cause de", "en raison de", "a provoqué", "a entraîné", "donc", "grâce à", "ce qui a"],
    "tr": ["nedeniyle", "yüzünden", "bu yüzden", "sonucunda", "neden oldu", "yol açtı", "sebep oldu",
           "dolayısıyla", "sayesinde", "için", "kaynaklandı", "tetikledi"],
}
LANG_NAMES = {"en": "English", "de": "German", "nl": "Dutch", "es": "Spanish", "fr": "French", "tr": "Turkish"}

# Non-causal relations between the SAME two phrases: hard negatives.
NEGATIVE_KINDS = [
    "purely temporal sequence with no causal link (use 'after', 'before', 'while', 'then')",
    "coincidence or correlation explicitly without causation",
    "a negated causal claim (the first did NOT cause the second)",
    "a conditional or hypothetical that has not happened (if/would)",
    "two unrelated facts joined by 'and' or listed together",
]


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.c = httpx.Client(base_url=base_url.rstrip("/"), timeout=90,
                              headers={"Authorization": f"Bearer {api_key}"})
        self.model = model

    def json(self, prompt: str, max_tokens: int = 1800) -> object:
        for attempt in range(6):
            body = {
                "model": self.model, "temperature": 0.9, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": "Reply with JSON only. No prose, no markdown fences."},
                             {"role": "user", "content": prompt}]}
            if "gpt-oss" in self.model:
                body["reasoning_effort"] = "low"   # reasoning tokens otherwise eat the budget -> empty content
            r = self.c.post("/chat/completions", json=body)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 2 * (attempt + 1)))); continue
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            text = text.split("</think>", 1)[-1].strip()
            if not text:
                max_tokens = min(max_tokens * 2, 8000); time.sleep(1); continue
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            self.last_raw = text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"[\[{].*[\]}]", text, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
                prompt = prompt + "\n\nIMPORTANT: output must be a single valid JSON array, nothing else."
                time.sleep(1)
        raise RuntimeError(f"LLM did not return JSON; last reply started: {self.last_raw[:200]!r}")


# English clauses embed unchanged ("because the child spilled juice"); in German,
# Turkish, Dutch etc. a finite clause changes word order or inflection inside a
# sentence, so the verbatim rule would reject almost everything. For those
# languages the phrases are NOMINALISED events (noun phrases), which drop into any
# position unchanged and are the natural news/report style anyway.
PHRASE_FORM = {
    "en": "a short self-contained event clause (4-10 words) that reads naturally after the word 'because': "
          "include articles and a subject with a verb (e.g. 'the child spilled juice on the carpet')",
}
PHRASE_FORM_NOMINAL = ("a short NOUN PHRASE naming the event (3-8 words), so it can be inserted unchanged anywhere "
                       "in a sentence: nominalise the verb (e.g. German 'der Ausfall des Servers am Montag', "
                       "Turkish 'sunucunun pazartesi çökmesi', Dutch 'de storing van de server op maandag')")


def seed_pairs(llm: LLM, domain: str, n: int, lang: str, form_name: str = "auto") -> list[dict]:
    if form_name == "nominal" or lang not in PHRASE_FORM:
        form, form_name = PHRASE_FORM_NOMINAL, "nominal"
    else:
        form, form_name = PHRASE_FORM[lang], "clause"
    prompt = (
        f"Write {n} realistic cause -> effect event pairs from {DOMAINS[domain]}, in {LANG_NAMES[lang]}. "
        f"Each side is {form}. Use the language's normal capitalisation rules, no pronouns, no trailing period. "
        "Vary subjects, avoid repeating nouns. "
        'Return a JSON list of objects {"cause": "...", "effect": "..."}.'
    )
    out = llm.json(prompt)
    pairs = []
    for p in out if isinstance(out, list) else out.get("pairs", []):
        c, e = (p.get("cause") or "").strip().rstrip("."), (p.get("effect") or "").strip().rstrip(".")
        if 2 <= len(c.split()) <= 12 and 2 <= len(e.split()) <= 12 and c.lower() != e.lower():
            pairs.append({"cause": c, "effect": e, "domain": domain, "lang": lang, "form": form_name})
    return pairs


def realize(llm: LLM, pairs: list[dict], per_pair: int, lang: str) -> list[dict]:
    """Ask for sentences containing both phrases verbatim; keep only exact hits.
    Each realisation gets a random register and causal style so the corpus is not
    one flavour of prose."""
    items = []
    for i, p in enumerate(pairs):
        for k in range(per_pair):
            styles = [s for j, s in enumerate(CAUSAL_STYLES) if p.get("form") == "nominal" or j not in NOMINAL_ONLY_STYLES]
            items.append({"id": i, "k": k, "cause": p["cause"], "effect": p["effect"],
                          "register": random.choice(list(REGISTERS)), "style": random.choice(styles)})
    reg_desc = "; ".join(f"{k}: {v}" for k, v in REGISTERS.items())
    prompt = (
        f"For each item write ONE natural {LANG_NAMES[lang]} text (8-35 words) stating that the cause led to the "
        "effect, in the requested register and causal style. HARD RULE: the text must contain the cause text and "
        "the effect text EXACTLY as given, character for character (same words, same order, same case, no "
        "inflection changes), each exactly once. Build the text around the phrases as fixed blocks. Never use merely "
        "temporal words (after, when, then, while) as the only link. Keep the cause as the cause.\n"
        f"Registers: {reg_desc}\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
        'Return a JSON list of objects {"id": <item id>, "k": <k>, "sentence": "..."}.'
    )
    out = llm.json(prompt, max_tokens=4500)
    rows = []
    by_key = {(it["id"], it["k"]): it for it in items}
    for o in out if isinstance(out, list) else out.get("sentences", []):
        try:
            it = by_key[(int(o["id"]), int(o.get("k", 0)))]
            p = pairs[int(o["id"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        s = (o.get("sentence") or "").strip()
        row = verify(s, p["cause"], p["effect"], lang)
        if row:
            row.update({"domain": p["domain"], "lang": lang, "kind": "causal", "form": p.get("form"),
                        "register": it["register"], "style": it["style"]})
            rows.append(row)
    return rows


def chains(llm: LLM, pairs: list[dict], lang: str) -> list[dict]:
    """Two-relation texts A -> B -> C built from pairs that share a phrase, or from a
    pair plus a fresh consequence. Emitted as CNC does it: one row per relation with
    the same text (num_rs=2)."""
    items = []
    for i, p in enumerate(pairs):
        items.append({"id": i, "a": p["cause"], "b": p["effect"]})
    prompt = (
        f"For each item, first invent a plausible further consequence c of b (a short {LANG_NAMES[lang]} "
        "phrase in the same form as a and b), then write ONE natural text (15-40 words, one or two sentences) "
        "stating that a led to b and b led to c. HARD RULE: a, b and c must each appear EXACTLY as given, once. "
        "Use two different connectives or one implicit link.\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
        'Return a JSON list of objects {"id": <id>, "c": "...", "sentence": "..."}.'
    )
    out = llm.json(prompt, max_tokens=4000)
    rows = []
    for o in out if isinstance(out, list) else out.get("items", []):
        try:
            p = pairs[int(o["id"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        s, c = (o.get("sentence") or "").strip(), (o.get("c") or "").strip().rstrip(".")
        if not c or s.count(c) != 1:
            continue
        r1 = verify(s, p["cause"], p["effect"], lang)
        r2 = verify(s, p["effect"], c, lang)
        if r1 and r2:
            for r, rel in ((r1, 1), (r2, 2)):
                r.update({"domain": p["domain"], "lang": lang, "kind": "causal", "register": "prose",
                          "style": "chain", "num_rs": 2, "rel": rel})
                rows.append(r)
    return rows


def plain_facts(llm: LLM, domain: str, n: int, lang: str) -> list[dict]:
    """Non-causal facts of the kind agents store most (the gate must stay quiet)."""
    prompt = (
        f"Write {n} varied {LANG_NAMES[lang]} statements from {DOMAINS[domain]} that an assistant might remember: "
        "preferences, states, schedules, numbers, decisions, descriptions. Mix registers (chat, note, prose, "
        "first person, log line). NONE may express or imply that one thing caused another, and none may use "
        "because/due to/led to/so that. 6-30 words each. Return a JSON list of strings."
    )
    out = llm.json(prompt, max_tokens=2500)
    rows = []
    for s in out if isinstance(out, list) else out.get("items", []):
        s = str(s).strip()
        if 5 <= len(s.split()) <= 40:
            rows.append({"text": s, "cause": None, "effect": None, "a": None, "b": None,
                         "domain": domain, "lang": lang, "kind": "negative", "register": "mixed", "style": "plain"})
    return rows


_TYPOS = {"the": "teh", "and": "adn", "because": "becuase", "with": "wiht", "their": "thier"}


def noise(row: dict, rng: random.Random) -> dict | None:
    """Span-preserving surface noise (typos outside spans, casing, dropped final period)
    on a copy of the row; None if nothing could be changed safely."""
    s = row["text"]
    protected = [tuple(row["cause_span"]), tuple(row["effect_span"])] if row["kind"] == "causal" else []
    if row.get("signal_span"):
        protected.append(tuple(row["signal_span"]))
    words = list(re.finditer(r"\b\w+\b", s))
    free = [m for m in words if not any(a <= m.start() < b for a, b in protected)]
    if not free:
        return None
    m = rng.choice(free)
    w = m.group(0)
    if w.lower() in _TYPOS:
        repl = _TYPOS[w.lower()]
    elif len(w) > 4:
        i = rng.randrange(1, len(w) - 1); repl = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    else:
        repl = w.upper() if rng.random() < 0.5 else w.lower()
    delta = len(repl) - len(w)
    new = s[:m.start()] + repl + s[m.end():]
    if rng.random() < 0.3 and new.endswith("."):
        new = new[:-1]
    shift = lambda span: [span[0] + (delta if span[0] > m.start() else 0), span[1] + (delta if span[1] > m.start() else 0)]  # noqa: E731
    out = dict(row, text=new, style=row.get("style", "") + "+noise")
    if row["kind"] == "causal":
        out["cause_span"], out["effect_span"] = shift(row["cause_span"]), shift(row["effect_span"])
        if row.get("signal_span"):
            out["signal_span"] = shift(row["signal_span"])
        if new[out["cause_span"][0]:out["cause_span"][1]] != row["cause"] or new[out["effect_span"][0]:out["effect_span"][1]] != row["effect"]:
            return None
    return out


def verify(sentence: str, cause: str, effect: str, lang: str) -> dict | None:
    if not (8 <= len(sentence.split()) <= 40):
        return None
    if sentence.count(cause) != 1 or sentence.count(effect) != 1:
        return None
    ci, ei = sentence.index(cause), sentence.index(effect)
    if not (ci + len(cause) <= ei or ei + len(effect) <= ci):
        return None
    # Word-aligned boundaries: the trainer splits on spaces, so a phrase glued to a
    # bracket or quote ("(the outage") would drag that character into the span.
    for start, end in ((ci, ci + len(cause)), (ei, ei + len(effect))):
        if start > 0 and not sentence[start - 1].isspace():
            return None
        if end < len(sentence) and (sentence[end].isalnum() or sentence[end] in "'\u2019"):
            return None
    # signal: longest lexicon connective outside both argument spans (case-insensitive)
    low = sentence.lower()
    sig = None
    for cand in sorted(SIGNALS.get(lang, SIGNALS["en"]), key=len, reverse=True):
        m = re.search(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", low)
        if m and not (ci <= m.start() < ci + len(cause)) and not (ei <= m.start() < ei + len(effect)):
            sig = (m.start(), m.end()); break
    return {"text": sentence, "cause": cause, "effect": effect,
            "cause_span": [ci, ci + len(cause)], "effect_span": [ei, ei + len(effect)],
            "signal_span": list(sig) if sig else None}


def causal_check(llm: LLM, rows: list[dict], lang: str) -> list[dict]:
    """Second-pass QA: an independent yes/no causal judgement; drop rows that disagree
    with their label (positives judged non-causal, negatives judged causal)."""
    keep = []
    for i in range(0, len(rows), 15):
        chunk = rows[i:i + 15]
        items = [{"id": j, "sentence": r["text"], "a": r.get("cause") or r.get("a"), "b": r.get("effect") or r.get("b")}
                 for j, r in enumerate(chunk)]
        prompt = ("For each sentence decide: (1) does it assert that one of the two phrases caused, led to, enabled "
                  "or was done in order to bring about the other (purpose counts as causal; mere sequence, "
                  "correlation, conditionals and negated causation do not)? If yes, which phrase is the CAUSE? "
                  "(2) Is the text grammatical and natural for its register (chat and notes may be terse, but no "
                  "broken syntax like 'made it possible the meeting started')?\n"
                  f"Items: {json.dumps(items, ensure_ascii=False)}\n"
                  'Return a JSON list of objects {"id": <id>, "causal": true|false, "cause": "a"|"b"|null, "grammatical": true|false}.')
        try:
            out = llm.json(prompt, max_tokens=1500)
            verdict = {int(o["id"]): (bool(o.get("causal")), o.get("cause"), o.get("grammatical", True))
                       for o in (out if isinstance(out, list) else out.get("items", []))}
        except Exception:
            keep += chunk; continue
        for j, r in enumerate(chunk):
            v = verdict.get(j)
            if v is None:
                keep.append(r); continue
            is_causal, cause_side, ok_grammar = v
            if not ok_grammar:
                continue
            if r["kind"] == "causal" and is_causal and cause_side in (None, "a"):
                keep.append(r)          # agrees, and direction not contradicted
            elif r["kind"] == "negative" and not is_causal:
                keep.append(r)
    return keep


def negatives(llm: LLM, pairs: list[dict], lang: str) -> list[dict]:
    items = [{"id": i, "a": p["cause"], "b": p["effect"], "relation": random.choice(NEGATIVE_KINDS)} for i, p in enumerate(pairs)]
    prompt = (
        f"For each item write one natural {LANG_NAMES[lang]} sentence (10-30 words) that mentions phrase a and phrase b "
        "EXACTLY as given, each once, but expresses the given non-causal relation between them. The sentence must NOT "
        "state or imply that a caused b.\n"
        f"Items: {json.dumps(items, ensure_ascii=False)}\n"
        'Return a JSON list of objects {"id": <item id>, "sentence": "..."}.'
    )
    out = llm.json(prompt, max_tokens=3000)
    rows = []
    for o in out if isinstance(out, list) else out.get("sentences", []):
        try:
            p = pairs[int(o["id"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        s = (o.get("sentence") or "").strip()
        if s.count(p["cause"]) == 1 and s.count(p["effect"]) == 1 and 8 <= len(s.split()) <= 40:
            rows.append({"text": s, "cause": None, "effect": None, "a": p["cause"], "b": p["effect"],
                         "domain": p["domain"], "lang": lang, "kind": "negative"})
    return rows


def to_text_w_pairs(row: dict) -> str:
    """CNC inline format: <ARG0>cause</ARG0>, <ARG1>effect</ARG1>, <SIG0>signal</SIG0>."""
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
    ap.add_argument("--domains", default="tech,support,business,personal,news")
    ap.add_argument("--langs", default="en")
    ap.add_argument("--pairs", type=int, default=20, help="seed pairs per domain per language")
    ap.add_argument("--per-pair", type=int, default=2)
    ap.add_argument("--negatives", type=float, default=0.3, help="fraction of pairs also realised as non-causal")
    ap.add_argument("--chains", type=float, default=0.15, help="fraction of pairs extended into a two-relation text")
    ap.add_argument("--plain", type=int, default=10, help="plain non-causal facts per domain per language")
    ap.add_argument("--noise", type=float, default=0.15, help="fraction of causal rows duplicated with surface noise")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    llm = LLM(os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"), os.environ["LLM_API_KEY"],
              os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {"pairs": 0, "asked": 0, "accepted": 0, "neg_asked": 0, "neg_accepted": 0}
    for lang in args.langs.split(","):
        for domain in args.domains.split(","):
            if lang in PHRASE_FORM:
                pairs = seed_pairs(llm, domain, args.pairs // 2, lang, "clause") + seed_pairs(llm, domain, args.pairs - args.pairs // 2, lang, "nominal")
            else:
                pairs = seed_pairs(llm, domain, args.pairs, lang)
            stats["pairs"] += len(pairs)
            for i in range(0, len(pairs), 10):
                chunk = pairs[i:i + 10]
                got = realize(llm, chunk, args.per_pair, lang)
                stats["asked"] += len(chunk) * args.per_pair; stats["accepted"] += len(got); rows += got
            neg_pairs = random.sample(pairs, max(1, int(len(pairs) * args.negatives))) if pairs else []
            for i in range(0, len(neg_pairs), 10):
                got = negatives(llm, neg_pairs[i:i + 10], lang)
                stats["neg_asked"] += len(neg_pairs[i:i + 10]); stats["neg_accepted"] += len(got); rows += got
            chain_pairs = random.sample(pairs, max(1, int(len(pairs) * args.chains))) if pairs else []
            for i in range(0, len(chain_pairs), 8):
                got = chains(llm, chain_pairs[i:i + 8], lang)
                stats["chain_rows"] = stats.get("chain_rows", 0) + len(got); rows += got
            if args.plain:
                got = plain_facts(llm, domain, args.plain, lang)
                stats["plain_facts"] = stats.get("plain_facts", 0) + len(got); rows += got
    before = len(rows)
    rows = causal_check(llm, rows, "en")
    stats["qa_dropped"] = before - len(rows)
    # dedupe by normalised text; keep pair id for leak-free splits
    seen, final = set(), []
    for r in rows:
        key = re.sub(r"\W+", " ", r["text"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        r["pair_id"] = hashlib.sha1(f"{r.get('cause')}|{r.get('effect')}|{r['domain']}".encode()).hexdigest()[:10]
        final.append(r)
    with (out / "synth.jsonl").open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "synth_cnc.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "text", "text_w_pairs", "num_rs", "lang", "domain", "pair_id"])
        for i, r in enumerate(final):
            twp = to_text_w_pairs(r) if r["kind"] == "causal" else r["text"]
            w.writerow([i, r["text"], twp, r.get("num_rs", 1) if r["kind"] == "causal" else 0, r["lang"], r["domain"], r["pair_id"]])
    rng = random.Random(args.seed)
    noised = [n for n in (noise(r, rng) for r in final if r["kind"] == "causal" and rng.random() < args.noise) if n]
    final += noised
    stats["final"] = len(final)
    stats["noised"] = len(noised)
    stats["with_signal"] = sum(1 for r in final if r.get("signal_span"))
    # distribution report: this is how you judge variation
    from collections import Counter
    causal_rows = [r for r in final if r["kind"] == "causal"]
    stats["registers"] = dict(Counter(r.get("register", "?") for r in causal_rows))
    stats["styles"] = dict(Counter((r.get("style") or "").replace("+noise", "")[:32] for r in causal_rows))
    stats["cause_first"] = sum(1 for r in causal_rows if r["cause_span"][0] < r["effect_span"][0])
    stats["effect_first"] = len(causal_rows) - stats["cause_first"]
    stats["implicit_no_signal"] = sum(1 for r in causal_rows if not r.get("signal_span"))
    stats["negatives_shared_phrase"] = sum(1 for r in final if r["kind"] == "negative" and r.get("a"))
    stats["negatives_plain"] = sum(1 for r in final if r["kind"] == "negative" and not r.get("a"))
    stats["len_words_p10_p50_p90"] = [sorted(len(r["text"].split()) for r in final)[int(q * (len(final) - 1))] for q in (0.1, 0.5, 0.9)]
    print(json.dumps(stats, indent=1))
    for r in random.sample(final, min(12, len(final))):
        tag = f"{r['kind']:8} {r.get('register','')[:6]:6} {(r.get('style') or '')[:18]:18}"
        print(f"[{tag}] {to_text_w_pairs(r) if r['kind']=='causal' else r['text']}")


if __name__ == "__main__":
    main()
