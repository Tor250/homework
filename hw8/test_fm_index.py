import random

from hw8.fm_index import FMIndex


def _naive_search(text: str, pattern: str):
    if not pattern:
        return list(range(len(text)))

    result = []
    width = len(pattern)
    for start in range(len(text) - width + 1):
        if text[start : start + width] == pattern:
            result.append(start)
    return result


def test_search_matches_naive_reference_on_random_text():
    rng = random.Random(42)
    alphabet = "abcd "
    text = "".join(rng.choice(alphabet) for _ in range(400))
    index = FMIndex(text, checkpoint_step=16)

    for _ in range(120):
        width = rng.randint(1, 6)
        if rng.random() < 0.65 and len(text) >= width:
            start = rng.randint(0, len(text) - width)
            pattern = text[start : start + width]
        else:
            pattern = "".join(rng.choice(alphabet) for _ in range(width))

        expected = _naive_search(text, pattern)
        assert index.search(pattern) == expected
        assert index.count(pattern) == len(expected)


def test_overlapping_occurrences_and_full_text_match():
    text = "aaaaaa"
    index = FMIndex(text)

    assert index.search("aaa") == [0, 1, 2, 3]
    assert index.search("aaaaaa") == [0]
    assert index.search("b") == []
    assert index.search("") == [0, 1, 2, 3, 4, 5]


def test_empty_text_is_safe():
    index = FMIndex("")

    assert index.search("a") == []
    assert index.search("") == []
    assert index.count("a") == 0


def test_rebuild_overwrites_previous_text():
    index = FMIndex("banana")
    assert index.search("ana") == [1, 3]

    index.build("abracadabra")
    assert index.search("abra") == [0, 7]
    assert index.count("cad") == 1
