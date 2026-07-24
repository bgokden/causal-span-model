"""Multilingual inference for the pointer model via script-aware segmentation.

The pointer model predicts spans over the "words" it is given. For whitespace
languages a word is a whitespace token; for CJK / Thai (no spaces) each character
is its own word, otherwise the whole sentence collapses into a single word and the
model returns the entire sentence for every span. ``segment`` handles both (and
mixed text) and records each word's character offsets, so extracted spans are
sliced straight out of the ORIGINAL text -- no space-joining artifacts in any
language. This is what makes the (stronger) pointer model usable everywhere, so no
separate multilingual model or model-routing is needed.
"""

import torch

from causal_span_model.pointer.decode import decode_relations


def _is_cjk(ch: str) -> bool:
    return (
        "一" <= ch <= "鿿"  # CJK unified ideographs
        or "぀" <= ch <= "ヿ"  # hiragana + katakana
        or "㐀" <= ch <= "䶿"  # CJK extension A
        or "가" <= ch <= "힣"  # hangul syllables
        or "฀" <= ch <= "๿"  # thai
    )


def segment(text: str) -> list[tuple[str, int, int]]:
    """Return [(word, char_start, char_end)]; CJK/Thai characters are single words."""
    words: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if _is_cjk(ch):
            words.append((ch, i, i + 1))
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and not _is_cjk(text[j]):
                j += 1
            words.append((text[i:j], i, j))
            i = j
    return words


def predict_relations(model, tokenizer, text: str, max_len: int = 256,
                      topk: int = 5, device: str = "cpu") -> list[dict]:
    """Return deduplicated ``[{'cause','effect','signal'}]`` for one text, any language.

    Spans are sliced from the original text via character offsets, so CJK output has
    no inserted spaces and Latin output keeps original spacing.
    """
    segments = segment(text)
    if not segments:
        return []
    words = [w for w, _, _ in segments]
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device))
    # Causal gate: if the model has a causal head and predicts non-causal, emit
    # nothing (the span head always produces spans, even on non-causal text).
    if "causal_cls" in out and int(out["causal_cls"][0].argmax().item()) == 0:
        return []
    length = enc["input_ids"].shape[1]
    logits = {k: out[k][0].cpu() for k in
              ("cause_start", "cause_end", "effect_start", "effect_end", "sig_start", "sig_end")}
    has_signal = bool(out["signal_cls"][0].argmax().item())
    decoded = decode_relations(logits, sep_pos=length - 1, has_signal=has_signal,
                               beam=True, topk=topk)

    def slice_span(span):
        if span is None:
            return ""
        start_word, end_word = word_ids[span[0]], word_ids[span[1]]
        if start_word is None or end_word is None:
            return ""
        return text[segments[start_word][1]:segments[end_word][2]].strip()

    relations: list[dict] = []
    for rel in decoded:
        cause = slice_span(rel["cause"])
        effect = slice_span(rel["effect"])
        if not cause or not effect:
            continue
        # Drop near-duplicates from beam top-2: same cause and one effect nested in
        # the other (a single-relation sentence yielding two boundary variants).
        if any(k["cause"].lower() == cause.lower()
               and (effect.lower() in k["effect"].lower() or k["effect"].lower() in effect.lower())
               for k in relations):
            continue
        relations.append({"cause": cause, "effect": effect, "signal": slice_span(rel["signal"])})
    return relations
