"""Causal span model: train a multilingual BIO tagger for cause/effect/signal.

A standalone training pipeline for the span model that reasongraph's
``OnnxTokenClassifierExtractor`` consumes. Only the model-free helpers are
re-exported here; training/export/eval pull in torch/transformers lazily via
their own modules.
"""

from causal_span_model.cnc_to_bio import iter_labeled_segments, tagged_to_bio
from causal_span_model.evaluate import decode_bio_spans, to_causal_relations
from causal_span_model.labels import (
    CNC_ROLE_TO_TYPE,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    SPAN_TYPES,
)
from causal_span_model.translate import (
    project_example,
    project_from_translation,
    wrap_with_sentinels,
)

__all__ = [
    "LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "SPAN_TYPES",
    "CNC_ROLE_TO_TYPE",
    "iter_labeled_segments",
    "tagged_to_bio",
    "wrap_with_sentinels",
    "project_from_translation",
    "project_example",
    "decode_bio_spans",
    "to_causal_relations",
]
