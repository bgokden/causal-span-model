"""Multilingual augmentation of BIO causal examples via span-preserving MT.

CNC is English-only, but reasongraph needs a multilingual span model. This step
projects the cause/effect/signal spans onto machine translations so the model
learns the same span types across languages.

Method: each labeled span is wrapped in a sentinel pair (``@@C@@ ... @@/C@@`` for
cause, ``E`` for effect, ``S`` for signal), the wrapped sentence is translated
with NLLB-200, and the sentinels are parsed back out of the translation to
recover spans in the target language. When the model drops, crosses, unbalances
or loses a required sentinel, or the run structure of the spans changes, the
example is DISCARDED rather than emitted with corrupt labels -- coverage is
traded for label quality, and the drop rate is reported.

Sentinel choice matters: the earlier ``⟦...⟧`` markers (U+27E6/U+27E7) are NOT in
NLLB's SentencePiece vocab -- they tokenize to <unk> and vanish in translation,
dropping 100% of examples. ``@@C@@`` round-trips verbatim through the NLLB
tokenizer. Post-translation survival is still model-dependent, so measure it on a
sample (``--sample``) before a full run.

The sentinel parsing and validation (``wrap_with_sentinels`` /
``project_from_translation`` / ``project_example``) are pure and model-free, so
they are unit-tested without downloading any translation model.
"""

import argparse
import collections
import json
import sys
import re

TYPE_TO_CODE = {"CAUSE": "C", "EFFECT": "E", "SIGNAL": "S"}
CODE_TO_TYPE = {code: span_type for span_type, code in TYPE_TO_CODE.items()}

# Sentinels that survive NLLB's SentencePiece tokenizer round-trip verbatim.
_OPEN_FMT = "@@{}@@"
_CLOSE_FMT = "@@/{}@@"
_SENT_RE = re.compile(r"@@(/?)([CES])@@")

# NLLB FLORES-200 codes for the shipped target languages (mirrors the reasongraph
# causal cue lexicon coverage).
DEFAULT_TARGET_LANGS = {
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "tr": "tur_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
}


def _sentinel(code: str, closing: bool = False) -> str:
    return _CLOSE_FMT.format(code) if closing else _OPEN_FMT.format(code)


def group_spans(tokens: list[str], tags: list[str]):
    """Yield ``(span_type_or_None, [tokens])`` runs, respecting B- boundaries.

    Two adjacent spans of the same type separated by a ``B-`` tag stay separate.
    """
    groups: list[tuple[str | None, list[str]]] = []
    cur_type: str | None = None
    cur: list[str] = []
    have = False
    for token, tag in zip(tokens, tags):
        if tag == "O":
            span_type, is_begin = None, False
        else:
            prefix, span_type = tag.split("-", 1)
            is_begin = prefix == "B"
        new_group = (not have) or (span_type != cur_type) or (is_begin and span_type is not None)
        if new_group:
            if have:
                groups.append((cur_type, cur))
            cur_type, cur, have = span_type, [token], True
        else:
            cur.append(token)
    if have:
        groups.append((cur_type, cur))
    return groups


def _run_type_counts(tokens: list[str], tags: list[str]) -> collections.Counter:
    """Multiset of labeled-span run types (order-independent, count-sensitive)."""
    return collections.Counter(
        span_type for span_type, _ in group_spans(tokens, tags) if span_type is not None
    )


def wrap_with_sentinels(tokens: list[str], tags: list[str]) -> str:
    """Rebuild a sentence string with each labeled span wrapped in sentinels."""
    parts: list[str] = []
    for span_type, span_tokens in group_spans(tokens, tags):
        if span_type is None:
            parts.extend(span_tokens)
        else:
            code = TYPE_TO_CODE[span_type]
            parts.append(_sentinel(code))
            parts.extend(span_tokens)
            parts.append(_sentinel(code, closing=True))
    return " ".join(parts)


def project_from_translation(translated: str):
    """Parse a sentinel-wrapped translation back into ``(tokens, tags)`` or ``None``.

    Returns ``None`` when the sentinels are malformed (dropped, crossed, nested
    or unbalanced by the translator), so a bad projection is discarded instead of
    producing corrupt labels.
    """
    # Force sentinels to be standalone tokens even if the translator glued them
    # to neighbouring words.
    spaced = _SENT_RE.sub(lambda m: f" {m.group(0)} ", translated)
    tokens: list[str] = []
    tags: list[str] = []
    open_type: str | None = None
    first = False
    for token in spaced.split():
        match = _SENT_RE.fullmatch(token)
        if match:
            closing = match.group(1) == "/"
            span_type = CODE_TO_TYPE[match.group(2)]
            if closing:
                if open_type != span_type:
                    return None  # crossed or unmatched close
                open_type = None
            else:
                if open_type is not None:
                    return None  # a second open before the first closed
                open_type = span_type
                first = True
            continue
        if open_type is None:
            tags.append("O")
        else:
            tags.append(f"{'B' if first else 'I'}-{open_type}")
            first = False
        tokens.append(token)
    if open_type is not None:
        return None  # a span was never closed
    if not tokens:
        return None
    return tokens, tags


def project_example(source_tokens: list[str], source_tags: list[str], translated: str):
    """Project one example; return ``(tokens, tags)`` or ``None`` if unreliable.

    Beyond the structural checks in :func:`project_from_translation`, this drops
    translations whose multiset of span-type runs differs from the source (a span
    lost, split or merged), which the type-set check alone would miss.
    """
    projected = project_from_translation(translated)
    if projected is None:
        return None
    projected_tokens, projected_tags = projected
    if _run_type_counts(source_tokens, source_tags) != _run_type_counts(projected_tokens, projected_tags):
        return None
    return projected


class NllbTranslator:
    """Lazy, batched NLLB-200 translator. Loads torch/transformers on first use."""

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        max_len: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.src_lang = src_lang
        self.max_len = max_len
        self.device = device
        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.src_lang)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def translate_batch(self, texts: list[str], tgt_lang: str) -> list[str]:
        """Translate a batch of texts into ``tgt_lang`` in a single generate call."""
        self._load()
        import torch

        if not texts:
            return []
        inputs = self._tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len,
        ).to(self.device)
        forced_bos = self._tok.convert_tokens_to_ids(tgt_lang)
        # Allow the target to be longer than the source (verbose languages) while
        # still bounding runaway generation. Separate from the encoder cap above.
        max_input = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=int(1.5 * max_input) + 16,
            )
        return self._tok.batch_decode(generated, skip_special_tokens=True)

    def translate(self, text: str, tgt_lang: str) -> str:
        return self.translate_batch([text], tgt_lang)[0]


def augment_jsonl(
    in_path: str,
    out_path: str,
    translator: NllbTranslator,
    target_langs: dict[str, str] | None = None,
    keep_source: bool = True,
    batch_size: int = 32,
    sample: int = 0,
) -> dict[str, tuple[int, int]]:
    """Read English BIO JSONL, append translated examples, write combined JSONL.

    Translation is batched per language (one generate call per ``batch_size``
    sentences). Returns per-language ``(kept, dropped)`` counts. English originals
    are kept when ``keep_source`` is set. When ``sample`` > 0, prints that many
    source/translation/projection triples per language for a fidelity eyeball.
    """
    langs = target_langs or DEFAULT_TARGET_LANGS
    examples: list[dict] = []
    with open(in_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    wrapped = [wrap_with_sentinels(ex["tokens"], ex["tags"]) for ex in examples]
    stats: dict[str, tuple[int, int]] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        if keep_source:
            for ex in examples:
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for lang, code in langs.items():
            kept = 0
            dropped = 0
            shown = 0
            for start in range(0, len(wrapped), batch_size):
                chunk = wrapped[start:start + batch_size]
                translations = translator.translate_batch(chunk, code)
                for offset, translated in enumerate(translations):
                    ex = examples[start + offset]
                    projected = project_example(ex["tokens"], ex["tags"], translated)
                    if sample and shown < sample:
                        print(f"[{lang}] SRC : {' '.join(ex['tokens'])}")
                        print(f"[{lang}] MT  : {translated}")
                        print(f"[{lang}] PROJ: {projected[0] if projected else 'DROPPED'}")
                        shown += 1
                    if projected is None:
                        dropped += 1
                        continue
                    p_tokens, p_tags = projected
                    out.write(json.dumps(
                        {"tokens": p_tokens, "tags": p_tags, "lang": lang},
                        ensure_ascii=False,
                    ) + "\n")
                    kept += 1
            stats[lang] = (kept, dropped)
            total = kept + dropped
            rate = (dropped / total * 100) if total else 0.0
            print(f"{lang}: kept {kept}, dropped {dropped} ({rate:.1f}% dropped)")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="English BIO JSONL from cnc_to_bio")
    parser.add_argument("output", help="combined multilingual BIO JSONL")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--langs", nargs="*", default=None,
                        help="subset of target language keys (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample", type=int, default=0,
                        help="print N source/translation/projection triples per language")
    parser.add_argument("--no-keep-source", action="store_true",
                        help="drop the English originals from the output")
    args = parser.parse_args(argv)

    targets = DEFAULT_TARGET_LANGS
    if args.langs:
        targets = {k: DEFAULT_TARGET_LANGS[k] for k in args.langs}
    translator = NllbTranslator(model_id=args.model)
    augment_jsonl(
        args.input, args.output, translator,
        target_langs=targets, keep_source=not args.no_keep_source,
        batch_size=args.batch_size, sample=args.sample,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
