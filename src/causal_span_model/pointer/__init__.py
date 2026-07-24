"""Span-pointer causal model (ST2-style) to beat the CNC Subtask-2 baseline.

A separate model family from the BIO tagger: instead of per-token BIO labels it
predicts 6 pointer logits per token ({start,end} x {cause,effect,signal}) and
decodes spans under ordering/non-overlap constraints, with beam search yielding
the top-2 relations per sentence. This is the architecture the CNC organizer
baseline uses; here it runs on microsoft/mdeberta-v3-base.

This model does NOT fit reasongraph's generic OnnxTokenClassifierExtractor -- it
needs its own decoder. It ships as a separate artifact.

Architecture adapted from tanfiona/CausalNewsCorpus (MIT).
"""
