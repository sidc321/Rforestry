import numpy as np
import pytest

from Rforestry import RandomForest
from helpers import get_data


@pytest.fixture
def forest():
    forest = RandomForest(
        ntree=500,
        replace=True,
        sample_fraction=0.8,
        mtry=3,
        nodesizeStrictSpl=5,
        splitrule="variance",
        splitratio=1,
        nodesizeStrictAvg=5,
        seed=2,
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_get_params(forest):
    assert forest.get_params()
    assert forest.get_params()["ntree"] == 500
    assert len(forest.get_params().keys()) == 26


def test_set_params(forest):
    X, y = get_data()

    forest.set_params(ntree=1000, maxDepth=5, seed=1729)
    assert forest.get_params()["ntree"] == 1000
    assert forest.get_params()["maxDepth"] == 5

    forest.fit(X, y)
    y1 = forest.predict(X)

    forest.set_params(seed=1)
    assert forest.get_params()["ntree"] == 1000
    assert forest.get_params()["maxDepth"] == 5

    forest.fit(X, y)
    y2 = forest.predict(X)

    assert not np.array_equal(y1, y2)
