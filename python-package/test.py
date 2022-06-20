
from hashlib import new
import numpy as np
import pandas as pd
import warnings
import math
import os
from random import randrange
import sys
from forestry import forestry
import Py_preprocessing

from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import platform
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from forestry_shadow import ShadowForestryTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import platform


data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
X = X.loc[:, X.columns != 'sepal length (cm)']
y = df['sepal length (cm)']


fr = forestry(
        ntree = 500,
        maxDepth=2,
        interactionDepth=2,
        verbose=False,
        scale=False,
        seed=1729
)


fr.fit(X, y)

fr.translate_tree_python(5)
shadow_forestry = ShadowForestryTree(fr, X, y, X.columns.values, 'sepal length (cm)', 1)

viz = dtreeviz(shadow_forestry,
                scale=3.0,
                target_name='sepal length (cm)',
                feature_names=X.columns.values)

viz.view()


# Sklearn

fr2 = RandomForestRegressor(
        n_estimators=500,
        max_depth=2,
        random_state=1729,
)

fr2.fit(X, y)

