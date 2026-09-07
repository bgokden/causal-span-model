"""Cause/effect swap augmentation for the span-pointer trainer (A1).

Takes pairs of tagged CNC relation strings that share a signal connective and swaps
one role's span between them, minting new tagged sentences with novel cause->effect
combinations under a real connective (e.g. keep "... <SIG0>because</SIG0> ..." but
substitute a different cause phrase). Ungrammatical results are dropped by an optional
LLM grammar check (the generator's QA), so no broken syntax enters training.

Spans in CNC never overlap and appear in a fixed left-to-right order, so a swap is a
clean word-range substitution followed by re-tagging.
"""

import json
import os
import urllib.request

from .data import word_bounds

_NAMES = ["ARG0", "ARG1", "SIG0"]  # cause, effect, signal


def _spans(starts, ends):
    out = []
    for k in range(3):
        if starts[k] != -100 and ends[k] != -100:
            out.append((starts[k], ends[k], _NAMES[k]))
    return sorted(out)


def _role_words(tokens, starts, ends, role):
    k = _NAMES.index(role)
    if starts[k] == -100:
        return None
    return tokens[starts[k]:ends[k] + 1]


def _signal_text(tokens, starts, ends):
    w = _role_words(tokens, starts, ends, "SIG0")
    return " ".join(w).lower() if w else "<none>"


def rebuild(tokens, starts, ends, replacements):
    """Serialize tokens back to a tagged string, substituting given role word-lists."""
    out, i = [], 0
    for s, e, name in _spans(starts, ends):
        out.extend(tokens[i:s])
        words = replacements.get(name) or tokens[s:e + 1]
        open_t, close_t = f"<{name}>", f"</{name}>"
        if len(words) == 1:
            out.append(open_t + words[0] + close_t)
        else:
            out.append(open_t + words[0])
            out.extend(words[1:-1])
            out.append(words[-1] + close_t)
        i = e + 1
    out.extend(tokens[i:])
    return " ".join(out)


def _clean(tagged):
    import re
    return re.sub(r"</?[A-Z]+\d*>", "", tagged)


class LLMGrammar:
    """Batched grammaticality check via an OpenAI-compatible endpoint (Ollama)."""

    def __init__(self, base_url=None, api_key=None, model=None):
        self.url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/") + "/chat/completions"
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "ollama")
        self.model = model or os.environ["LLM_MODEL"]

    def __call__(self, sentences):
        ok = [True] * len(sentences)
        for i in range(0, len(sentences), 15):
            chunk = sentences[i:i + 15]
            items = [{"id": j, "sentence": s} for j, s in enumerate(chunk)]
            prompt = ("For each sentence decide only whether it is grammatical and natural "
                      "(no broken syntax like 'made it possible the meeting started'). "
                      f"Items: {json.dumps(items, ensure_ascii=False)}\n"
                      'Return a JSON list of {"id": <id>, "grammatical": true|false}.')
            body = {"model": self.model, "temperature": 0,
                    "messages": [{"role": "system", "content": "Reply with JSON only."},
                                 {"role": "user", "content": prompt}]}
            if "gpt-oss" in self.model:
                body["reasoning_effort"] = "low"
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    content = json.loads(resp.read())["choices"][0]["message"]["content"]
                content = content.split("</think>", 1)[-1].strip()
                start, end = content.find("["), content.rfind("]")
                verdict = {int(o["id"]): bool(o.get("grammatical", True))
                           for o in json.loads(content[start:end + 1])}
                for j in range(len(chunk)):
                    ok[i + j] = verdict.get(j, True)
            except Exception:
                pass  # on QA failure keep the batch (conservative); never silently drop all
        return ok


def swap_augment(rels, n, seed=42, grammar_check=None, roles=("ARG0", "ARG1")):
    """Return up to ``n`` new tagged strings from same-signal cause/effect swaps.

    ``grammar_check`` is a callable(list[str]) -> list[bool] over the CLEAN sentences;
    non-grammatical swaps are dropped. With no check, all swaps are kept.
    """
    import random

    parsed = []
    for tw in rels:
        try:
            tokens, starts, ends = word_bounds(tw)
        except Exception:
            continue
        if starts[0] == -100 or starts[1] == -100:  # need cause and effect
            continue
        parsed.append((tokens, starts, ends))

    groups = {}
    for idx, (tokens, starts, ends) in enumerate(parsed):
        groups.setdefault(_signal_text(tokens, starts, ends), []).append(idx)

    rng = random.Random(seed)
    candidates, seen = [], set()
    # several shuffled pairing rounds to expand the candidate pool past one pass
    rounds = max(1, (n // max(1, len(parsed) // 2)) + 2)
    for _ in range(rounds):
        for sig, idxs in groups.items():
            if len(idxs) < 2:
                continue
            order = list(idxs)
            rng.shuffle(order)
            for a, b in zip(order[::2], order[1::2]):
                ta, sa, ea = parsed[a]
                tb, sb, eb = parsed[b]
                for role in roles:
                    donor = _role_words(tb, sb, eb, role)
                    if not donor:
                        continue
                    new = rebuild(ta, sa, ea, {role: donor})
                    if new not in seen and "<ARG0>" in new and "<ARG1>" in new:
                        seen.add(new)
                        candidates.append(new)
    rng.shuffle(candidates)

    kept = []
    batch = 300
    for start in range(0, len(candidates), batch):
        if len(kept) >= n:
            break
        chunk = candidates[start:start + batch]
        if grammar_check is not None:
            flags = grammar_check([_clean(c) for c in chunk])
            chunk = [c for c, ok in zip(chunk, flags) if ok]
        for c in chunk:
            kept.append(c)
            if len(kept) >= n:
                break
    return kept[:n]
