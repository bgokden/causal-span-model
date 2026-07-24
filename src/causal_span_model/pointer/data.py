"""Data prep for the span-pointer model.

Turns a CNC tagged relation string into clean tokens plus start/end SUBWORD
indices for cause/effect/signal. One training example per relation. Word-level
bounds follow the corpus tokenization (single-space split, tags attached); those
word indices are then mapped to subword-token indices (first subword for a start,
last subword for an end), matching the baseline's alignment.
"""

import re
from ast import literal_eval

import pandas as pd
from datasets import Dataset

_TAG = re.compile(r"</?[A-Z]+\d*>")


def clean_tok(token: str) -> str:
    return _TAG.sub("", token)


def word_bounds(text_w_pairs: str):
    """Return (clean_tokens, starts[3], ends[3]) as WORD indices.

    starts/ends are [cause, effect, signal]; signal is -100 when absent.
    """
    cause: list[int] = []
    effect: list[int] = []
    signal: list[int] = []
    tokens: list[str] = []
    for i, token in enumerate(text_w_pairs.split(" ")):
        if "<ARG0>" in token:
            cause.append(i)
        if "</ARG0>" in token:
            cause.append(i)
        if "<ARG1>" in token:
            effect.append(i)
        if "</ARG1>" in token:
            effect.append(i)
        if "<SIG" in token:  # opening signal tag (</SIG..> does not contain '<SIG')
            signal.append(i)
            if "</SIG" in token:  # single-word signal
                signal.append(i)
        elif "</SIG" in token:
            signal.append(i)
        tokens.append(clean_tok(token))

    starts = [cause[0], effect[0]]
    ends = [cause[1], effect[1]]
    if len(signal) >= 2:
        starts.append(signal[0])
        ends.append(signal[1])
    else:
        starts.append(-100)
        ends.append(-100)
    return tokens, starts, ends


def _align(tokens, starts, ends, tokenizer, max_len):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_len)
    word_ids = enc.word_ids()
    word2tok: dict[int, list[int]] = {}
    for token_pos, word_id in enumerate(word_ids):
        if word_id is not None:
            word2tok.setdefault(word_id, []).append(token_pos)
    max_word = max(word2tok) if word2tok else 0

    def start_idx(word: int):
        if word == -100:
            return -100
        w = word
        while w not in word2tok and w > 0:
            w -= 1
        toks = word2tok.get(w)
        return toks[0] if toks else -100

    def end_idx(word: int):
        if word == -100:
            return -100
        w = word
        while w not in word2tok and w < max_word:
            w += 1
        toks = word2tok.get(w)
        return toks[-1] if toks else -100

    sp = [start_idx(starts[k]) for k in range(3)]
    ep = [end_idx(ends[k]) for k in range(3)]
    enc["start_positions"] = sp
    enc["end_positions"] = ep
    return enc


def build_dataset(rows, tokenizer, max_len: int = 256) -> Dataset:
    """Build a tokenized dataset from an iterable of tagged relation strings.

    Drops any example whose cause/effect span was lost to truncation (index
    -100), so no example trains on a mutilated target.
    """
    features = {"input_ids": [], "attention_mask": [], "start_positions": [],
                "end_positions": [], "causal_labels": []}
    dropped = 0
    for text_w_pairs in rows:
        tokens, starts, ends = word_bounds(text_w_pairs)
        enc = _align(tokens, starts, ends, tokenizer, max_len)
        sp, ep = enc["start_positions"], enc["end_positions"]
        if -100 in sp[:2] or -100 in ep[:2]:  # cause/effect must survive
            dropped += 1
            continue
        features["input_ids"].append(enc["input_ids"])
        features["attention_mask"].append(enc["attention_mask"])
        features["start_positions"].append(sp)
        features["end_positions"].append(ep)
        features["causal_labels"].append(1)
    if dropped:
        print(f"[pointer-data] dropped {dropped} examples (cause/effect truncated)")
    return Dataset.from_dict(features)


def build_negatives(texts, tokenizer, max_len: int = 256) -> Dataset:
    """Build non-causal examples (no spans, ``causal_labels`` = 0) for the gate."""
    features = {"input_ids": [], "attention_mask": [], "start_positions": [],
                "end_positions": [], "causal_labels": []}
    for text in texts:
        words = str(text).split(" ")
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len)
        features["input_ids"].append(enc["input_ids"])
        features["attention_mask"].append(enc["attention_mask"])
        features["start_positions"].append([-100, -100, -100])
        features["end_positions"].append([-100, -100, -100])
        features["causal_labels"].append(0)
    return Dataset.from_dict(features)


def non_causal_texts(csv_path: str) -> list[str]:
    """Return the plain texts of non-causal sentences (num_rs == 0) from a grouped CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "num_rs" not in df.columns:
        return []
    return [str(t) for t in df.loc[df["num_rs"] == 0, "text"] if isinstance(t, str) and t.strip()]


def relations_from_csv(csv_path: str) -> list[str]:
    """Extract one tagged relation string per causal relation from a CNC CSV.

    Handles both the per-relation ``text_w_pairs`` column and the grouped
    ``causal_text_w_pairs`` (stringified list) column.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    out: list[str] = []
    if "causal_text_w_pairs" in df.columns:
        for value in df["causal_text_w_pairs"]:
            if not isinstance(value, str) or not value.strip() or value.strip() == "nan":
                continue
            for rel in literal_eval(value):
                if isinstance(rel, str) and "<ARG0>" in rel and "<ARG1>" in rel:
                    out.append(rel)
    else:
        for value in df.get("text_w_pairs", []):
            if isinstance(value, str) and "<ARG0>" in value and "<ARG1>" in value:
                out.append(value)
    return out
