import os
from re import T
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'Rforestry'))

from sklearn.datasets import load_iris
from forestry import RandomForest

import numpy as np
import pandas as pd

import pytest

def get_data():
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    return X, y


@pytest.fixture
def get_forest():
    forest = RandomForest(seed = 432432)

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_tree_features(get_forest):
    X, y = get_data()

    with pytest.raises(ValueError):
        p = get_forest.predict(newdata=X, trees=np.array([1,2,3,4,4,500]))

    try:
        p = get_forest.predict(newdata=X, trees=np.array([-500,2,3,4,4,499]))
    except ValueError:
        assert False

    with pytest.raises(ValueError):
        p = get_forest.predict(newdata=X, trees=np.array([-501,2,3,4,4,500]))

    with pytest.raises(ValueError):
        p = get_forest.predict(newdata=X, trees=[1,2,3,4,4,499], aggregation='oob')

    with pytest.raises(ValueError):
        p = get_forest.predict(newdata=X, trees=[1,2,3,4,4,499], aggregation='average', exact=False)

def test_predict_settings(get_forest):
    X, y = get_data()

    p1 = get_forest.predict(newdata=X)
    p2 = get_forest.predict(newdata=X, trees = np.arange(500))
    assert np.array_equal(p1, p2) == True

def test_linearity(get_forest):
    X, y = get_data()

    p1 = get_forest.predict(newdata=X, trees = [1])
    p2 = get_forest.predict(newdata=X, trees = [2])
    p3 = get_forest.predict(newdata=X, trees = [3])
    p4 = get_forest.predict(newdata=X, trees = [4])

    p_all = get_forest.predict(newdata=X, trees = [1,2,3,4])
    p_agg = 0.25*(p1 + p2 + p3 + p4)

    assert np.array_equal(p_all, p_agg) == True


    p_all = get_forest.predict(newdata=X, trees = [1,1,1,2,2])
    p_agg = (p1*3 + p2*2)/5

    assert np.array_equal(p_all, p_agg) == True
