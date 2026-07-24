"""The span-pointer model: encoder + 6-channel start/end pointer head.

Head channels, in order: cause-start, cause-end, effect-start, effect-end,
signal-start, signal-end (ARG0=cause, ARG1=effect, SIG=signal). Each channel is a
per-token logit; a span is a (start, end) pair chosen by the decoder. Training
targets are token indices; -100 marks an absent signal (cause/effect always
present in a causal relation) and is ignored by the loss.
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

CHANNELS = 6


class PointerCausalModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, CHANNELS)
        # Binary "does this relation have a signal span?" head on the [CLS] token,
        # so inference only decodes a signal when one is predicted to exist.
        self.signal_classifier = nn.Linear(self.config.hidden_size, 2)
        # Binary "is this text causal?" gate on [CLS]; lets inference emit nothing
        # on non-causal text (the span head otherwise always produces spans).
        self.causal_classifier = nn.Linear(self.config.hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        causal_labels=None,
        **kwargs,
    ) -> dict:
        sequence_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [B, L, 6]
        channels = [c.squeeze(-1).contiguous() for c in logits.split(1, dim=-1)]
        cause_start, cause_end, effect_start, effect_end, sig_start, sig_end = channels
        pooled = sequence_output[:, 0, :]
        signal_cls = self.signal_classifier(pooled)  # [B, 2]
        causal_cls = self.causal_classifier(pooled)   # [B, 2]
        result = {
            "cause_start": cause_start, "cause_end": cause_end,
            "effect_start": effect_start, "effect_end": effect_end,
            "sig_start": sig_start, "sig_end": sig_end,
            "signal_cls": signal_cls, "causal_cls": causal_cls,
        }

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()  # ignore_index=-100 by default
            zero = torch.zeros((), device=input_ids.device)

            def _safe(value):  # a whole batch may have no valid targets (all -100)
                return zero if torch.isnan(value) else value

            cause = _safe((loss_fct(cause_start, start_positions[:, 0])
                           + loss_fct(cause_end, end_positions[:, 0])) / 2)
            effect = _safe((loss_fct(effect_start, start_positions[:, 1])
                            + loss_fct(effect_end, end_positions[:, 1])) / 2)
            sig = _safe((loss_fct(sig_start, start_positions[:, 2])
                         + loss_fct(sig_end, end_positions[:, 2])) / 2)
            has_signal_label = (end_positions[:, 2] != -100).long()
            terms = [cause, effect, sig, loss_fct(signal_cls, has_signal_label)]
            if causal_labels is not None:
                terms.append(loss_fct(causal_cls, causal_labels))
            result["loss"] = sum(terms) / len(terms)

        return result
