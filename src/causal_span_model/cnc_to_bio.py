"""Convert Causal News Corpus (CNC) span annotations into BIO training data.

CNC subtask-2 ships each causal relation as a sentence whose cause/effect/signal
spans are marked inline with ``<ARG0>...</ARG0>`` (cause), ``<ARG1>...</ARG1>``
(effect) and ``<SIG0>...</SIG0>`` (signal) tags in the ``text_w_pairs`` column.
A sentence with two causal relations appears as two rows; each row becomes one
BIO example, which is the standard CNC subtask-2 setup.

Output is JSON Lines, one object per line::

    {"tokens": ["Heavy", "rain", "caused", "flooding"],
     "tags":   ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"],
     "lang":   "en"}

CNC tags can NEST -- a signal is frequently marked inside an argument span, e.g.
``<ARG0><SIG0>to</SIG0> foil the march</ARG0>``. The parser therefore keeps a
stack and labels each token with the INNERMOST active span (signal wins over the
enclosing argument), so the argument tokens around the signal keep their label
instead of collapsing to O. Roles seen in CNC V2: ARG0, ARG1, SIG0, SIG1, SIG2.

Data source: https://github.com/tanfiona/CausalNewsCorpus (CC0-1.0).
"""

import argparse
import collections
import csv
import json
import re
import sys

from causal_span_model.labels import CNC_ROLE_TO_TYPE

_TAG_RE = re.compile(r"</?(ARG\d+|SIG\d+)>")

# Zero-width characters that would otherwise ride along on a token and produce a
# word that tokenizes to no subwords (silently losing its label downstream).
_ZERO_WIDTH = {ord(c): None for c in "​‌‍﻿"}


def _clean_word(word: str) -> str:
    return word.translate(_ZERO_WIDTH)


def _role_to_type(role: str) -> str | None:
    """Map a CNC role (ARG0/ARG1/SIG*) to a span type, or None to ignore it."""
    mapped = CNC_ROLE_TO_TYPE.get(role)
    if mapped is not None:
        return mapped
    if role.startswith("SIG"):
        return "SIGNAL"  # SIG2+ are still signals
    return None  # unexpected ARG2+ -> transparent (kept balanced on the stack)


def iter_labeled_segments(text_w_pairs: str):
    """Yield ``(segment_text, span_type_or_None)`` over a CNC tagged string.

    Walks the inline tags with a stack so nested tags resolve to the innermost
    active span type. A close tag pops the most recent matching role, returning
    to the enclosing span rather than to O. Robust to tags glued to words or
    separated by whitespace, because it slices on tag character offsets.
    """
    stack: list[tuple[str, str | None]] = []

    def active_type() -> str | None:
        for _, span_type in reversed(stack):
            if span_type is not None:
                return span_type
        return None

    pos = 0
    for match in _TAG_RE.finditer(text_w_pairs):
        segment = text_w_pairs[pos:match.start()]
        if segment:
            yield segment, active_type()
        role = match.group(1)
        if match.group(0).startswith("</"):
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == role:
                    del stack[i]
                    break
        else:
            stack.append((role, _role_to_type(role)))
        pos = match.end()
    tail = text_w_pairs[pos:]
    if tail:
        yield tail, active_type()


def tagged_to_bio(text_w_pairs: str) -> tuple[list[str], list[str]]:
    """Convert one CNC tagged string into parallel ``tokens`` and ``tags`` lists."""
    tokens: list[str] = []
    tags: list[str] = []
    for segment, span_type in iter_labeled_segments(text_w_pairs):
        first = True
        for raw in segment.split():
            word = _clean_word(raw)
            if not word:
                continue  # was purely zero-width; drop it and its (absent) label
            if span_type is None:
                tags.append("O")
            else:
                tags.append(f"{'B' if first else 'I'}-{span_type}")
                first = False
            tokens.append(word)
    return tokens, tags


def _is_missing(value: str | None) -> bool:
    return value is None or value.strip() == "" or value.strip().lower() == "nan"


def convert_csv(
    in_path: str,
    out_path: str,
    text_col: str = "text_w_pairs",
    plain_col: str = "text",
    lang: str = "en",
    include_negatives: bool = False,
    relations: str = "all",
) -> tuple[int, int]:
    """Convert a CNC CSV to a BIO JSONL file. Returns ``(rows_read, rows_written)``.

    A sentence with several causal relations appears as several rows (``eg_id``
    0, 1, ...), each with its own span labelling. Those rows give the SAME input
    tokens different labels, which is contradictory training signal for a
    single-label tagger. ``relations`` controls the policy:

      * ``"all"`` (default): keep every relation row -- more data, the standard
        CNC subtask-2 setup, at the cost of some contradictory labels.
      * ``"primary"``: keep only the first relation (``eg_id == 0``) per sentence
        -- consistent labels, fewer examples.

    The fraction of multi-relation sentences is logged either way. Rows without a
    tagged relation are skipped unless ``include_negatives`` is set.
    """
    # utf-8-sig strips the BOM CNC ships on its first column name.
    with open(in_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    per_sentence = collections.Counter(
        (row.get("doc_id"), row.get("sent_id"))
        for row in rows
        if not _is_missing(row.get(text_col))
    )
    multi = sum(1 for count in per_sentence.values() if count > 1)
    if per_sentence:
        pct = multi / len(per_sentence) * 100
        print(f"[cnc] {multi}/{len(per_sentence)} sentences ({pct:.0f}%) have multiple "
              f"relations; policy={relations}")

    rows_read = 0
    rows_written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            rows_read += 1
            tagged = row.get(text_col)
            if _is_missing(tagged):
                if not include_negatives:
                    continue
                plain = row.get(plain_col)
                if _is_missing(plain):
                    continue
                tokens = plain.split()
                tags = ["O"] * len(tokens)
            else:
                if relations == "primary" and str(row.get("eg_id", "0")) != "0":
                    continue
                tokens, tags = tagged_to_bio(tagged)
            if not tokens:
                continue
            out.write(json.dumps(
                {"tokens": tokens, "tags": tags, "lang": row.get("lang", lang)},
                ensure_ascii=False,
            ) + "\n")
            rows_written += 1
    return rows_read, rows_written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CNC CSV with a text_w_pairs column")
    parser.add_argument("output", help="destination BIO JSONL file")
    parser.add_argument("--text-col", default="text_w_pairs")
    parser.add_argument("--plain-col", default="text")
    parser.add_argument("--lang", default="en", help="language tag for the rows")
    parser.add_argument("--include-negatives", action="store_true",
                        help="emit non-causal sentences as all-O examples")
    parser.add_argument("--relations", choices=["all", "primary"], default="all",
                        help="keep every relation row, or only the first per sentence")
    args = parser.parse_args(argv)

    read, written = convert_csv(
        args.input, args.output,
        text_col=args.text_col, plain_col=args.plain_col,
        lang=args.lang, include_negatives=args.include_negatives,
        relations=args.relations,
    )
    print(f"read {read} rows, wrote {written} BIO examples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
