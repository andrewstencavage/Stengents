import pytest

from normalize_index import normalize_index


class ItemsThatMustNotBeIndexedAtTheUpperBound:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError(f"unexpected item access at index {index}")


def test_rejects_index_at_the_upper_bound():
    with pytest.raises(IndexError):
        normalize_index(ItemsThatMustNotBeIndexedAtTheUpperBound(), 1)
