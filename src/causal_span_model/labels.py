"""Canonical BIO label scheme for the causal span model.

Single source of truth shared by conversion, training, export and evaluation so
the ``id2label`` written into ``config.json`` matches what every stage expects,
and matches what the reasongraph consumer decodes.

Direction is encoded in the label TYPE: CAUSE spans are the cause, EFFECT spans
are the effect, SIGNAL spans are the causal connective. Encoding direction in the
tags is exactly what lets the consumer read cause->effect straight from the model
output instead of relying on a separate cue lexicon for orientation.

The label ORDER is fixed and must not be reordered: the integer ids are baked
into every exported ``config.json`` and any change silently invalidates old
checkpoints.
"""

SPAN_TYPES = ["CAUSE", "EFFECT", "SIGNAL"]


def _build_labels(span_types: list[str]) -> list[str]:
    labels = ["O"]
    for span_type in span_types:
        labels.append(f"B-{span_type}")
        labels.append(f"I-{span_type}")
    return labels


LABELS = _build_labels(SPAN_TYPES)
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# Causal News Corpus argument roles -> our span types. ARG0 is the cause, ARG1
# the effect, SIG0/SIG1 the causal signal (connective).
CNC_ROLE_TO_TYPE = {
    "ARG0": "CAUSE",
    "ARG1": "EFFECT",
    "SIG0": "SIGNAL",
    "SIG1": "SIGNAL",
}
