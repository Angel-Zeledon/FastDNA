from fastdna.spectrum import suggest_min_count


def test_suggest_min_count_finds_a_known_valley():
    # A synthetic two-peak spectrum: a large error peak descending from
    # depth 1, a valley bottom at depth 5 (count 500), then a coverage peak
    # rising to depth 9 before falling off again.
    spectrum = {
        1: 10_000,
        2: 6_000,
        3: 3_000,
        4: 1_000,
        5: 500,
        6: 800,
        7: 2_000,
        8: 5_000,
        9: 8_000,
        10: 6_000,
        11: 3_000,
        12: 1_000,
        13: 500,
    }

    assert suggest_min_count(spectrum) == 5


def test_suggest_min_count_falls_back_on_a_monotonic_spectrum():
    # Strictly decreasing: no coverage peak visible in the sample, so no
    # valley exists to find.
    spectrum = {1: 1_000, 2: 800, 3: 600, 4: 400, 5: 200}
    assert suggest_min_count(spectrum) == 2


def test_suggest_min_count_falls_back_on_a_sparse_spectrum():
    assert suggest_min_count({1: 100, 2: 50}) == 2
    assert suggest_min_count({}) == 2


def test_suggest_min_count_respects_a_custom_default():
    spectrum = {1: 1_000, 2: 800, 3: 600}
    assert suggest_min_count(spectrum, default=7) == 7


def test_suggest_min_count_differs_per_sample():
    # The whole point: two samples with different coverage depths get
    # different thresholds, not a shared universal constant.
    shallow = {1: 9_000, 2: 4_000, 3: 1_000, 4: 300, 5: 1_500, 6: 3_000, 7: 1_000}
    deep = {1: 9_000, 2: 4_000, 3: 1_000, 4: 200, 5: 100, 6: 300, 7: 2_000, 8: 6_000, 9: 3_000}

    assert suggest_min_count(shallow) != suggest_min_count(deep)
