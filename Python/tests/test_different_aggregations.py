import pytest
from helpers import get_data

from Rforestry import RandomForest


def test_predict_error():
    X, y = get_data()

    forest = RandomForest(oob_honest=True)
    forest.fit(X, y)

    with pytest.raises(ValueError):
        forest.predict(aggregation="average")
