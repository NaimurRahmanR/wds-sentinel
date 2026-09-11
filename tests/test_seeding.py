import numpy as np

from wds_sentinel.utils.seeding import seed_everything


def test_same_seed_gives_identical_numpy_output():
    seed_everything(123)
    a = np.random.rand(10)
    seed_everything(123)
    b = np.random.rand(10)
    assert np.array_equal(a, b)


def test_different_seed_gives_different_output():
    seed_everything(1)
    a = np.random.rand(10)
    seed_everything(2)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)
