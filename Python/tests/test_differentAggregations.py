import pytest

from Rforestry import RandomForest
from helpers import get_data


def test_predict_error():
    X, y = get_data()

    rf = RandomForest(OOBhonest=True)
    rf.fit(X, y)

    with pytest.raises(ValueError):
        rf.predict(aggregation="average")
