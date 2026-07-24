from causal_span_model.cnc_to_bio import iter_labeled_segments, tagged_to_bio


def test_tagged_to_bio_basic_roles():
    tokens, tags = tagged_to_bio(
        "<ARG0>Heavy rainfall</ARG0> <SIG0>caused</SIG0> <ARG1>severe flooding</ARG1> ."
    )
    assert tokens == ["Heavy", "rainfall", "caused", "severe", "flooding", "."]
    assert tags == ["B-CAUSE", "I-CAUSE", "B-SIGNAL", "B-EFFECT", "I-EFFECT", "O"]


def test_tagged_to_bio_reversed_roles_keep_direction():
    # ARG1 (effect) precedes ARG0 (cause): direction stays encoded in the types.
    tokens, tags = tagged_to_bio(
        "<ARG1>The crash</ARG1> <SIG0>resulted from</SIG0> <ARG0>brake failure</ARG0> ."
    )
    assert tokens == ["The", "crash", "resulted", "from", "brake", "failure", "."]
    assert tags == ["B-EFFECT", "I-EFFECT", "B-SIGNAL", "I-SIGNAL", "B-CAUSE", "I-CAUSE", "O"]


def test_tagged_to_bio_handles_tags_glued_to_words():
    tokens, tags = tagged_to_bio("<ARG0>rain</ARG0><SIG0>caused</SIG0><ARG1>flood</ARG1>")
    assert tokens == ["rain", "caused", "flood"]
    assert tags == ["B-CAUSE", "B-SIGNAL", "B-EFFECT"]


def test_tagged_to_bio_nested_signal_in_arg():
    # Real CNC pattern: the signal is nested inside the cause argument. The
    # argument tokens around the signal must keep CAUSE, not collapse to O.
    tokens, tags = tagged_to_bio("<ARG0><SIG0>to</SIG0> foil the march</ARG0> .")
    assert tokens == ["to", "foil", "the", "march", "."]
    assert tags == ["B-SIGNAL", "B-CAUSE", "I-CAUSE", "I-CAUSE", "O"]


def test_tagged_to_bio_signal_splits_arg_span():
    # A signal in the middle of a span splits it into two labeled runs.
    tokens, tags = tagged_to_bio("<ARG1>heavy <SIG0>rain</SIG0> today</ARG1>")
    assert tokens == ["heavy", "rain", "today"]
    assert tags == ["B-EFFECT", "B-SIGNAL", "B-EFFECT"]


def test_tagged_to_bio_handles_sig2():
    tokens, tags = tagged_to_bio("<ARG0>x</ARG0> <SIG2>then</SIG2> <ARG1>y</ARG1>")
    assert tokens == ["x", "then", "y"]
    assert tags == ["B-CAUSE", "B-SIGNAL", "B-EFFECT"]


def test_iter_labeled_segments_states():
    segments = list(iter_labeled_segments("a <ARG0>b c</ARG0> d"))
    types = [span_type for _, span_type in segments]
    assert types == [None, "CAUSE", None]


def test_untagged_text_is_all_o():
    tokens, tags = tagged_to_bio("The market closed higher today .")
    assert tags == ["O"] * len(tokens)
