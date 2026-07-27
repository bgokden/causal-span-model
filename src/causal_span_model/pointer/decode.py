"""Constrained span decoding for the pointer model (per example, subword indices).

Cause and effect must be ordered and non-overlapping; the decoder tries both
orderings and picks the higher-scoring boundary, then fills the remaining
endpoints by argmax within the allowed region. Beam search keeps the top-2
boundary configurations -> up to two relations per sentence (the corpus has up to
a few relations; the scorer aligns them to the gold count). Signal is decoded
separately by argmax, capped at 5 tokens. Ported from the CNC baseline (MIT).
"""

import torch

_NEG = -1e9


def _prep(logit: torch.Tensor, sep_pos: int) -> torch.Tensor:
    """Mask [CLS]/[SEP] and return log-probabilities (PAD already excluded)."""
    x = logit.clone().float()
    x[0] = _NEG
    x[sep_pos] = _NEG
    return torch.log_softmax(x, dim=-1)


def _best_pivot(a: torch.Tensor, b: torch.Tensor):
    """argmax over i<j of a[i]+b[j]; returns (i, j, score)."""
    length = a.shape[0]
    outer = a[:, None] + b[None, :]
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=a.device), diagonal=1)
    outer = outer.masked_fill(~upper, float("-inf"))
    flat = int(outer.argmax())
    i, j = flat // length, flat % length
    return i, j, float(outer[i, j])


def decode_cause_effect_greedy(cause_start, cause_end, effect_start, effect_end):
    before_i, before_j, before = _best_pivot(cause_end, effect_start)   # cause ends / effect starts
    after_i, after_j, after = _best_pivot(effect_end, cause_start)       # effect ends / cause starts
    if before >= after:
        end_cause, start_effect = before_i, before_j
        sc = cause_start.clone(); sc[end_cause + 1:] = _NEG
        start_cause = int(sc.argmax())
        ee = effect_end.clone(); ee[:start_effect] = _NEG
        end_effect = int(ee.argmax())
    else:
        end_effect, start_cause = after_i, after_j
        ce = cause_end.clone(); ce[:start_cause] = _NEG
        end_cause = int(ce.argmax())
        es = effect_start.clone(); es[end_effect + 1:] = _NEG
        start_effect = int(es.argmax())
    return [(start_cause, end_cause, start_effect, end_effect)]


def decode_cause_effect_beam(cause_start, cause_end, effect_start, effect_end, topk: int = 5):
    length = cause_start.shape[0]
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=cause_start.device), diagonal=1)
    before = (cause_end[:, None] + effect_start[None, :]).masked_fill(~upper, float("-inf"))
    after = (effect_end[:, None] + cause_start[None, :]).masked_fill(~upper, float("-inf"))

    pivots = []
    for matrix, direction in ((before, "before"), (after, "after")):
        flat = matrix.flatten()
        valid = int((flat > float("-inf")).sum())
        if valid == 0:
            continue
        values, idxs = flat.topk(min(topk, valid))
        for value, index in zip(values.tolist(), idxs.tolist()):
            pivots.append((value, direction, index // length, index % length))
    pivots.sort(key=lambda x: x[0], reverse=True)

    full: dict[tuple, float] = {}
    for score, direction, i, j in pivots[:topk]:
        if direction == "before":
            end_cause, start_effect = i, j
            sc = cause_start.clone(); sc[end_cause + 1:] = _NEG
            sv, si = sc.topk(min(topk, end_cause + 1))
            ee = effect_end.clone(); ee[:start_effect] = _NEG
            ev, ei = ee.topk(min(topk, length - start_effect))
            for m in range(len(si)):
                for n in range(len(ei)):
                    key = (int(si[m]), end_cause, start_effect, int(ei[n]))
                    full[key] = score + float(sv[m]) + float(ev[n])
        else:
            end_effect, start_cause = i, j
            ce = cause_end.clone(); ce[:start_cause] = _NEG
            ev, ei = ce.topk(min(topk, length - start_cause))
            es = effect_start.clone(); es[end_effect + 1:] = _NEG
            sv, si = es.topk(min(topk, end_effect + 1))
            for m in range(len(ei)):
                for n in range(len(si)):
                    key = (start_cause, int(ei[m]), int(si[n]), end_effect)
                    full[key] = score + float(ev[m]) + float(sv[n])

    ranked = sorted(full.items(), key=lambda x: x[1], reverse=True)
    return [key for key, _ in ranked[:2]]


def decode_signal(sig_start, sig_end, sep_pos: int, max_span: int = 5):
    ss = sig_start.clone().float()
    ss[0] = _NEG
    ss[sep_pos] = _NEG
    start = int(ss.argmax())
    se = sig_end.clone().float()
    se[:start] = _NEG
    if start + max_span < se.shape[0]:
        se[start + max_span:] = _NEG
    end = int(se.argmax())
    return start, end


def decode_relations(logits: dict, sep_pos: int, has_signal: bool,
                     beam: bool = True, topk: int = 5) -> list[dict]:
    """Return a list of relations; each is {'cause': (s,e), 'effect': (s,e), 'signal': (s,e)|None}.

    Arguments are returned exactly as the model decodes them (this feeds the CNC
    submission scorer, whose gold tolerates a connective inside an argument, so no
    boundary cleanup is applied here). Downstream consumers that want connective-free
    spans post-process in ``infer.predict_relations``.
    """
    cs = _prep(logits["cause_start"], sep_pos)
    ce = _prep(logits["cause_end"], sep_pos)
    es = _prep(logits["effect_start"], sep_pos)
    ee = _prep(logits["effect_end"], sep_pos)
    tuples = (decode_cause_effect_beam(cs, ce, es, ee, topk) if beam
              else decode_cause_effect_greedy(cs, ce, es, ee))

    signal = None
    if has_signal:
        signal = decode_signal(logits["sig_start"], logits["sig_end"], sep_pos)

    relations = []
    for start_cause, end_cause, start_effect, end_effect in tuples:
        relations.append({
            "cause": (start_cause, end_cause),
            "effect": (start_effect, end_effect),
            "signal": signal,
        })
    return relations
