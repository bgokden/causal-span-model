import pytest

from causal_span_model.translate import (
    _SENT_RE,
    project_example,
    project_from_translation,
    wrap_with_sentinels,
)


def test_wrap_then_project_round_trips():
    tokens = ["Heavy", "rain", "caused", "flooding"]
    tags = ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"]
    wrapped = wrap_with_sentinels(tokens, tags)
    projected = project_from_translation(wrapped)
    assert projected == (tokens, tags)


def test_wrap_uses_ascii_sentinels():
    wrapped = wrap_with_sentinels(["a", "b"], ["B-CAUSE", "B-EFFECT"])
    assert "@@C@@" in wrapped and "@@/C@@" in wrapped and "@@E@@" in wrapped
    assert "⟦" not in wrapped  # the old <unk> sentinels are gone


def test_project_handles_glued_sentinels():
    # A translator that glued sentinels to neighbouring words still parses.
    projected = project_from_translation("@@C@@Heavy rain@@/C@@ @@S@@caused@@/S@@ @@E@@flooding@@/E@@")
    assert projected == (
        ["Heavy", "rain", "caused", "flooding"],
        ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"],
    )


def test_project_rejects_unclosed_span():
    assert project_from_translation("@@C@@ Heavy rain") is None


def test_project_rejects_crossed_close():
    assert project_from_translation("@@C@@ Heavy @@/E@@") is None


def test_project_rejects_nested_open():
    assert project_from_translation("@@C@@ @@E@@ x @@/E@@ @@/C@@") is None


def test_project_example_drops_when_span_type_lost():
    tokens = ["Heavy", "rain", "caused", "flooding"]
    tags = ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"]
    # The signal sentinel was dropped by the translator: a required type is gone.
    lost_signal = "@@C@@ Heavy rain @@/C@@ caused @@E@@ flooding @@/E@@"
    assert project_example(tokens, tags, lost_signal) is None


def test_project_example_drops_when_run_count_changes():
    # One cause span in the source, but the projection has the cause split in two.
    tokens = ["heavy", "rain", "today"]
    tags = ["B-CAUSE", "I-CAUSE", "I-CAUSE"]
    split = "@@C@@ heavy @@/C@@ rain @@C@@ today @@/C@@"
    assert project_example(tokens, tags, split) is None


def test_project_example_keeps_faithful_translation():
    tokens = ["Heavy", "rain", "caused", "flooding"]
    tags = ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"]
    faithful = "@@C@@ Fuertes lluvias @@/C@@ @@S@@ causaron @@/S@@ @@E@@ inundaciones @@/E@@"
    projected = project_example(tokens, tags, faithful)
    assert projected == (
        ["Fuertes", "lluvias", "causaron", "inundaciones"],
        ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"],
    )


def test_nllb_sentinels_survive_tokenizer_roundtrip():
    # The critical guard: sentinels must NOT be <unk> in NLLB's vocab (the old
    # bracket sentinels were, silently dropping 100% of augmented examples).
    pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    except Exception as e:  # noqa: BLE001 - model not cached / offline
        pytest.skip(f"nllb tokenizer unavailable: {e}")

    wrapped = wrap_with_sentinels(
        ["Heavy", "rain", "caused", "flooding"],
        ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT"],
    )
    ids = tok(wrapped)["input_ids"]
    assert tok.unk_token_id not in ids
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert len(_SENT_RE.findall(decoded)) == 6  # 3 open + 3 close survive
