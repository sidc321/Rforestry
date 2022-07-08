import os
from re import T
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Rforestry'))

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
    forest = RandomForest(
    ntree = 500,
    replace = True,
    sample_fraction=0.8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    splitrule = 'variance',
    splitratio = 1,
    nodesizeStrictAvg = 5,
    seed = 2
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest

def test_get_params(get_forest):
    assert get_forest.get_params()
    assert get_forest.get_params()['ntree'] == 500
    assert len(get_forest.get_params().keys()) == 26

def test_set_params(get_forest):
    X, y = get_data()

    get_forest.set_params(ntree = 1000, maxDepth=5, seed=1729)
    assert get_forest.get_params()['ntree'] == 1000
    assert get_forest.get_params()['maxDepth'] == 5

    get_forest.fit(X, y)
    y1 = get_forest.predict(X)

    get_forest.set_params(seed=1)
    assert get_forest.get_params()['ntree'] == 1000
    assert get_forest.get_params()['maxDepth'] == 5
    get_forest.fit(X, y)
    y2 = get_forest.predict(X)

    assert not np.array_equal(y1, y2)