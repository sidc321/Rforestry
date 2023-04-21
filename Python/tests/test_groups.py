import numpy as np
from helpers import get_data
from pandas import Series
from random_forestry import RandomForest


def test_groups():
    X, y = get_data()

    forest = RandomForest(ntree=1)
    forest.fit(X, y)
    pred = forest.predict(X, aggregation="average")

    forest = RandomForest(ntree=1)
    groups = Series([i for i in range(1, len(X) // 2 + 1) for _ in range(2)])
    forest.fit(X, y, groups = groups)
    pred_groups = forest.predict(X, aggregation="average")

    assert np.array_equal(pred, pred_groups)


def test_groups_oob():
    X, y = get_data()

    forest = RandomForest(ntree=1, oob_honest=True)
    groups = Series([i for i in range(1, len(X) // 2 + 1) for _ in range(2)])
    forest.fit(X, y, groups = groups)
    pred = forest.predict(X, aggregation="oob", return_weight_matrix = True)
    assert len(pred["predictions"]) == len(X)
