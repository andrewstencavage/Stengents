import pytest

from normalize_index import normalize_index


def test_rejects_index_at_the_upper_bound():
    with pytest.raises(IndexError):
        normalize_index(["a"], 1)
