import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'Rforestry'))
from sklearn.datasets import load_iris
from forestry import RandomForest

import numpy as np
import pandas as pd

import pytest


def test_different_predictions():
    
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    X = df.loc[:, df.columns != 'target']
    y = df['target']

    forest1 = RandomForest(ntree=1, maxDepth = 2, seed=1, scale=False)
    forest1.fit(X, y)
    p1 = forest1.predict(X)

    forest2 = RandomForest(ntree=1, maxDepth = 2, seed=1, scale=False)
    forest2.fit(X, y)
    p2 = forest2.predict(X)
    assert np.array_equal(p1, p2)

    data2 = load_iris()
    df2 = pd.DataFrame(data2['data'], columns=data2['feature_names'])
    df2['target'] = data2['target']
    X2 = df2.loc[:, df2.columns != 'target']
    y2 = df2['target']

    forest3 = RandomForest(ntree=1, maxDepth = 2, seed=1, scale=False)
    forest3.fit(X2, y2)
    p3 = forest3.predict(X2)

    assert np.array_equal(p1, p3)

