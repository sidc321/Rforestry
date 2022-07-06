import os
from re import T
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris
from forestry import RandomForest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import pytest

def test_bootstrap_intervals():
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    fr = RandomForest(seed=3242, OOBhonest=True)
    fr.fit(X_train, y_train)

    preds = fr.get_ci(newdata=X_test, level=0.99, method='OOB-bootstrap')
    assert np.sum((y_test < preds['CI.upper']) & (y_test > preds['CI.lower'])) != 0
    
