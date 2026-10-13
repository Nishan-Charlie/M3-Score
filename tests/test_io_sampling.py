import pytest

from m3score import protocol
from m3score.io.sampling import sample_indices


def test_everything_is_used_when_n_covers_the_set():
    assert sample_indices(10, 10, None, 0) == list(range(10))
    assert sample_indices(10, 50, None, 0) == list(range(10))
    assert sample_indices(10, None, None, 0) == list(range(10))


def test_default_cap_applies_when_n_is_omitted():
    assert len(sample_indices(protocol.MAX_IMAGES + 10, None, None, 0)) == protocol.MAX_IMAGES


def test_a_subset_is_random_seeded_and_never_the_head_of_the_list():
    first = sample_indices(1000, 100, None, 0)
    assert len(set(first)) == 100
    assert first == sample_indices(1000, 100, None, 0)
    assert first != sample_indices(1000, 100, None, 1)
    assert first != list(range(100))
    assert max(first) > 500


def test_known_subjects_are_spread_across_the_sample():
    # 100 subjects x 5 items, listed subject by subject as a sorted folder would be.
    # Taking the head of this list would return 20 subjects; the finding in
    # docs/REFERENCE_COHORT_FINDING.md is that this is not a valid reference.
    groups = [f"subject{s:03d}" for s in range(100) for _ in range(5)]
    chosen = sample_indices(500, 100, groups, 0)
    assert len({groups[i] for i in chosen}) == 100


def test_known_subjects_are_balanced():
    groups = ["a"] * 50 + ["b"] * 50 + ["c"] * 50 + ["d"] * 2
    chosen = sample_indices(152, 40, groups, 0)
    counts = {g: sum(groups[i] == g for i in chosen) for g in "abcd"}
    assert sum(counts.values()) == 40
    assert counts["d"] == 2
    assert max(counts[g] for g in "abc") - min(counts[g] for g in "abc") <= 1


def test_grouped_sampling_is_seeded():
    groups = [f"s{i % 7}" for i in range(70)]
    assert sample_indices(70, 20, groups, 3) == sample_indices(70, 20, groups, 3)
    assert sample_indices(70, 20, groups, 3) != sample_indices(70, 20, groups, 4)


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        sample_indices(10, 0, None, 0)


def test_groups_must_cover_every_item():
    with pytest.raises(ValueError, match="group labels"):
        sample_indices(10, 5, ["a"] * 9, 0)
