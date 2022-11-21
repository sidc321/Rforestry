import os
from re import T
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'Rforestry'))

from sklearn.datasets import load_iris
from forestry import RandomForest

import numpy as np
import pandas as pd

import pytest


def test_predict_error():
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    X = df.loc[:, df.columns != 'target']
    y = df['target']

    rf = RandomForest(OOBhonest=True)
    rf.fit(X, y)

    with pytest.raises(ValueError):
        rf.predict(aggregation = 'average')