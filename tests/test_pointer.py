import torch

from causal_span_model.pointer.data import word_bounds
from causal_span_model.pointer.decode import (
    decode_cause_effect_greedy,
    decode_signal,
)


def _peak(length, position, value=5.0):
    tensor = torch.zeros(length)
    tensor[position] = value
    return tensor


def test_word_bounds_roles():
    tokens, starts, ends = word_bounds(
        "<ARG0>Heavy rainfall</ARG0> <SIG0>caused</SIG0> <ARG1>severe flooding</ARG1> ."
    )
    assert tokens == ["Heavy", "rainfall", "caused", "severe", "flooding", "."]
    # word indices: cause 0-1, effect 3-4, signal 2-2
    assert starts == [0, 3, 2]
    assert ends == [1, 4, 2]


def test_word_bounds_no_signal_is_sentinel():
    _, starts, ends = word_bounds("<ARG1>The crash</ARG1> from <ARG0>brake failure</ARG0> .")
    assert starts[2] == -100 and ends[2] == -100  # signal absent
    assert starts[:2] == [3, 0] and ends[:2] == [4, 1]  # effect 0-1, cause 3-4


def test_greedy_cause_before_effect_ordering():
    length = 6
    tuples = decode_cause_effect_greedy(
        _peak(length, 1), _peak(length, 2),  # cause start/end
        _peak(length, 4), _peak(length, 5),  # effect start/end
    )
    assert tuples == [(1, 2, 4, 5)]


def test_greedy_effect_before_cause_ordering():
    length = 6
    tuples = decode_cause_effect_greedy(
        _peak(length, 4), _peak(length, 5),  # cause start/end (later)
        _peak(length, 1), _peak(length, 2),  # effect start/end (earlier)
    )
    start_cause, end_cause, start_effect, end_effect = tuples[0]
    # effect fully precedes cause, no overlap, start<=end within each
    assert end_effect < start_cause
    assert start_cause <= end_cause and start_effect <= end_effect


def test_segment_whitespace_cjk_and_mixed():
    from causal_span_model.pointer.infer import segment

    text = "Heavy rainfall caused"
    seg = segment(text)
    assert [w for w, _, _ in seg] == ["Heavy", "rainfall", "caused"]
    assert text[seg[0][1]:seg[1][2]] == "Heavy rainfall"  # char offsets slice back

    assert [w for w, _, _ in segment("暴雨导致洪灾")] == list("暴雨导致洪灾")
    assert [w for w, _, _ in segment("AI 导致 change")] == ["AI", "导", "致", "change"]


def test_signal_span_capped_at_five_tokens():
    length = 20
    sig_start = _peak(length, 3)
    sig_end = _peak(length, 15)  # far past the 5-token cap
    start, end = decode_signal(sig_start, sig_end, sep_pos=length - 1, max_span=5)
    assert start == 3
    assert end < start + 5  # capped, so the far peak at 15 is unreachable
