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


def test_suggest_min_count_ignores_a_noise_blip_on_the_descent():
    # A single-count uptick (30_001 > 30_000) partway down the error peak's
    # tail, well before the real valley at depth 8 (600) and the real
    # coverage peak at depth 11 (9_000). A naive "stop at the first
    # non-decrease" walk stops at depth 2 -- this must not happen: the blip
    # is nowhere near a real second peak (it's not even double the previous
    # depth's count), so it must be walked past.
    spectrum = {
        1: 50_000,
        2: 30_000,
        3: 30_001,
        4: 20_000,
        5: 10_000,
        6: 5_000,
        7: 800,
        8: 600,
        9: 2_000,
        10: 6_000,
        11: 9_000,
        12: 5_000,
        13: 2_000,
    }
    assert suggest_min_count(spectrum) == 8


def test_suggest_min_count_falls_back_when_no_real_peak_follows_a_blip():
    # Low-coverage data: overall descending, with one tiny uptick (301 >
    # 300) that is not followed by any genuine coverage peak -- the walk
    # never climbs back to double any candidate floor. A confidently wrong
    # threshold (the blip's own depth) is worse than the documented
    # fallback, so this must return `default`, not 4.
    spectrum = {1: 9_000, 2: 4_000, 3: 1_000, 4: 300, 5: 301, 6: 200, 7: 150, 8: 120, 9: 100}
    assert suggest_min_count(spectrum) == 2


def test_suggest_min_count_treats_a_missing_depth_as_zero():
    # Depth 4 is entirely absent (zero distinct k-mers there -- routine in
    # a small sample), while depths 3 and 5 are both present. Treating 3
    # and 5 as adjacent (as plain `sorted(spectrum)` iteration would) hides
    # the true, lower floor at depth 4 and reports the wrong valley (5).
    # The true floor is the missing depth 4 (implicitly zero), which must
    # be detected as depth 4, not depth 5.
    spectrum = {1: 1_000, 2: 600, 3: 300, 5: 100, 6: 250, 7: 600, 8: 1_200}
    assert suggest_min_count(spectrum) == 4
