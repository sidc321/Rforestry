import os
from re import T
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Python_package'))
print(sys.path)

from sklearn.datasets import load_iris
from forestry import forestry

import numpy as np
import pandas as pd

import pytest

def test_conformal_intervals():
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    
    X_train = X.iloc[0:124, :]
    y_train = y.iloc[0:124]
    X_test = X.iloc[124:, :]
    y_test = y.iloc[124:]

    fr = forestry(OOBhonest=True, seed=3242)
    fr.fit(X_train, y_train)

    preds = fr.getCI(newdata=X_test, level=0.95, method='OOB-conformal')
    assert np.sum((y_test < preds['CI.upper']) & (y_test > preds['CI.lower'])) != 0