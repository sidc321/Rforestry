import numpy as np
from helpers import get_data
from pandas import Series

from random_forestry import RandomForest


def test_groups():
    X, y = get_data()

    forest = RandomForest()
    forest.fit(X, y)
    pred_avg = forest.predict(X, aggregation="average")
    pred_oob = forest.predict(X, aggregation="oob")

    forest = RandomForest()
    groups = Series([i for i in range(len(X) // 10) for _ in range(10)])
    forest.fit(X, y, groups=groups)
    pred_avg_groups = forest.predict(X, aggregation="average")
    pred_oob_groups = forest.predict(X, aggregation="oob")

    assert np.array_equal(pred_avg, pred_avg_groups)
    assert not np.array_equal(pred_oob, pred_oob_groups)


def test_groups_honest():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    forest.fit(X, y)
    pred_avg = forest.predict(X, aggregation="average")
    pred_oob = forest.predict(X, aggregation="oob")

    forest = RandomForest(oob_honest=True)
    groups = Series([i for i in range(len(X) // 10) for _ in range(10)])
    forest.fit(X, y, groups=groups)
    pred_avg_groups = forest.predict(X, aggregation="average")
    pred_oob_groups = forest.predict(X, aggregation="oob")

    assert not np.array_equal(pred_avg, pred_avg_groups)
    assert not np.array_equal(pred_oob, pred_oob_groups)
