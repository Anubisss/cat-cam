import numpy as np
import pytest

from catcam import cats


@pytest.fixture(scope="session")
def real_cats():
    """The prepared cats shipped with the repo."""
    return [cats.load(name) for name in cats.available()]


@pytest.fixture
def rng():
    return np.random.default_rng(0)
