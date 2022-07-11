import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'Rforestry'))
from sklearn.datasets import load_iris
from forestry import RandomForest

import numpy as np
import pandas as pd

import pytest



data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
y = df['target']

forest = RandomForest(ntree=1, maxDepth = 2, seed=1)
forest.fit(X, y)
pred = forest.predict(X)

forest_scaled = RandomForest(ntree=1, maxDepth = 2, scale=True, seed=1)
forest_scaled.fit(X, y)
pred_scaled = forest_scaled.predict(X)



def test_different_predictions():
    assert np.array_equal(pred, pred_scaled) == True

