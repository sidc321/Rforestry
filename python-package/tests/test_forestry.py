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
    nthread = 1,
    splitrule = 'variance',
    splitratio = 1,
    nodesizeStrictAvg = 5,
    seed = 2
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest

@pytest.fixture
def get_predictions(get_forest):
    X, y = get_data()
    return get_forest.predict(X)


def test_newdata_wrong_dim(get_forest):
    X, y = get_data()
    ncol = X.shape[1]
    X = X.iloc[:, 0:ncol-1]
    
    with pytest.raises(ValueError):
        assert get_forest.predict(X)

def test_newdata_shuffled_warning(get_forest):
    X, y = get_data()
    with pytest.warns(UserWarning):
        get_forest.predict(X.iloc[:, ::-1])

def test_equal_predictions(get_forest):
    X, y = get_data()
    p1 = get_forest.predict(X)
    p2 = get_forest.predict(X.iloc[:, ::-1])

    assert np.array_equal(p1, p2) == True

def test_error(get_predictions):
    X, y = get_data()
    print(np.mean((get_predictions - y)**2))
    assert np.mean((get_predictions - y)**2) < 1
