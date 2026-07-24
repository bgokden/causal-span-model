"""Load BIO JSONL into a HuggingFace dataset and align labels to subword tokens.

The tokenizer splits words into subwords; only the first subword of each word
carries the word's label, and the rest (plus special tokens) get ``-100`` so the
loss and seqeval ignore them. This is the standard token-classification alignment
and keeps B-/I- boundaries meaningful.

``max_length`` truncation would silently drop the labels of any word past the
cutoff (and CNC puts the effect at the sentence tail, so truncation trains a
cause with no effect). :func:`drop_overlong` removes over-length examples up front
so nothing is silently mutilated, and reports how many were dropped.
"""

import json

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from causal_span_model.labels import LABEL2ID


def load_bio_jsonl(path: str) -> Dataset:
    """Read a BIO JSONL file into a dataset with tokens, integer labels and lang."""
    tokens_list: list[list[str]] = []
    labels_list: list[list[int]] = []
    lang_list: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            tokens_list.append(example["tokens"])
            labels_list.append([LABEL2ID[tag] for tag in example["tags"]])
            lang_list.append(example.get("lang", "en"))
    return Dataset.from_dict({"tokens": tokens_list, "labels": labels_list, "lang": lang_list})


def drop_overlong(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, max_len: int = 256) -> Dataset:
    """Drop examples whose subword length exceeds ``max_len`` (would be truncated).

    Prevents silent label truncation by removing the example entirely rather than
    training on a clipped label sequence. Reports the count.
    """
    original = len(dataset)

    def _fits(example) -> bool:
        ids = tokenizer(example["tokens"], is_split_into_words=True, add_special_tokens=True)["input_ids"]
        return len(ids) <= max_len

    kept = dataset.filter(_fits)
    dropped = original - len(kept)
    if dropped:
        print(f"[dataset] dropped {dropped}/{original} examples over {max_len} subwords "
              f"(avoids silent label truncation)")
    return kept


def tokenize_and_align(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 256,
) -> Dataset:
    """Tokenize word lists and align word labels to subword positions.

    Assumes :func:`drop_overlong` has already removed over-length examples;
    ``truncation`` here is a defensive backstop only.
    """

    def _map(batch):
        encoded = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_len,
        )
        aligned_labels: list[list[int]] = []
        for index, labels in enumerate(batch["labels"]):
            word_ids = encoded.word_ids(batch_index=index)
            previous = None
            row: list[int] = []
            for word_id in word_ids:
                if word_id is None:
                    row.append(-100)
                elif word_id != previous:
                    row.append(labels[word_id])
                else:
                    row.append(-100)
                previous = word_id
            aligned_labels.append(row)
        encoded["labels"] = aligned_labels
        return encoded

    return dataset.map(_map, batched=True, remove_columns=dataset.column_names)
