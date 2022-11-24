import numpy as np

from Rforestry import RandomForest
from helpers import get_data


def test_different_predictions():

    X, y = get_data()

    forest1 = RandomForest(ntree=10, maxDepth=2, seed=1, scale=False)
    forest1.fit(X, y)
    p1 = forest1.predict(X)

    forest2 = RandomForest(ntree=10, maxDepth=2, seed=1, scale=False)
    forest2.fit(X, y)
    p2 = forest2.predict(X)

    assert np.array_equal(p1, p2)

    X, y = get_data()

    forest3 = RandomForest(ntree=10, maxDepth=2, seed=1, scale=False)
    forest3.fit(X, y)
    p3 = forest3.predict(X)

    assert np.array_equal(p1, p3)
