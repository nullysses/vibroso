from scripts.generate import apply_stop_to_output, truncate_at_stop


def test_truncate_at_stop_without_match():
    text, stopped = truncate_at_stop("hello world", ["<|end|>"])

    assert text == "hello world"
    assert stopped is False


def test_truncate_at_stop_removes_stop_by_default():
    text, stopped = truncate_at_stop("answer<|end|>extra", ["<|end|>"])

    assert text == "answer"
    assert stopped is True


def test_truncate_at_stop_can_keep_stop():
    text, stopped = truncate_at_stop("answer<|end|>extra", ["<|end|>"], keep_stop=True)

    assert text == "answer<|end|>"
    assert stopped is True


def test_truncate_at_stop_uses_earliest_stop():
    text, stopped = truncate_at_stop("a<stop2>b<stop1>c", ["<stop1>", "<stop2>"])

    assert text == "a"
    assert stopped is True


def test_apply_stop_to_output_ignores_stop_in_prompt_when_prompt_matches():
    output, stopped = apply_stop_to_output(
        "prompt <|end|> generated <|end|> extra",
        "prompt <|end|> ",
        ["<|end|>"],
    )

    assert output == "prompt <|end|> generated "
    assert stopped is True
