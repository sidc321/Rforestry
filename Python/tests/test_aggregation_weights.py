import numpy as np
import pytest

from Rforestry import RandomForest
from helpers import get_data


@pytest.fixture
def forest():
    forest = RandomForest(seed=432432)

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_tree_features(forest):
    X, _ = get_data()

    with pytest.raises(ValueError):
        forest.predict(newdata=X, trees=np.array([1, 2, 3, 4, 4, 500]))

    try:
        forest.predict(newdata=X, trees=np.array([-500, 2, 3, 4, 4, 499]))
    except ValueError:
        assert False

    with pytest.raises(ValueError):
        forest.predict(newdata=X, trees=np.array([-501, 2, 3, 4, 4, 500]))

    with pytest.raises(ValueError):
        forest.predict(newdata=X, trees=[1, 2, 3, 4, 4, 499], aggregation="oob")

    with pytest.raises(ValueError):
        forest.predict(
            newdata=X, trees=[1, 2, 3, 4, 4, 499], aggregation="average", exact=False
        )


def test_predict_settings(forest):
    X, _ = get_data()

    p1 = forest.predict(newdata=X)
    p2 = forest.predict(newdata=X, trees=np.arange(500))
    assert np.array_equal(p1, p2)


def test_linearity(forest):
    X, _ = get_data()

    p1 = forest.predict(newdata=X, trees=np.array([1]))
    p2 = forest.predict(newdata=X, trees=[2])
    p3 = forest.predict(newdata=X, trees=[3])
    p4 = forest.predict(newdata=X, trees=[4])

    p_all = forest.predict(newdata=X, trees=[1, 2, 3, 4])
    p_agg = 0.25 * (p1 + p2 + p3 + p4)

    assert np.array_equal(p_all, p_agg)

    p_all = forest.predict(newdata=X, trees=[1, 1, 1, 2, 2])
    p_agg = (p1 * 3 + p2 * 2) / 5

    assert np.array_equal(p_all, p_agg)
