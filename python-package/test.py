
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



data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
X = X.loc[:, X.columns != 'sepal length (cm)']
y = df['sepal length (cm)']

fr = forestry(
        ntree = 500,
        maxDepth=2,
        verbose=False,
        scale=False,
        seed=1729
)

print("Fitting the forest")
fr.fit(X, y)

fr.translate_tree_python()
print(fr.Py_forest)


shadow_forestry = ShadowForestryTree(fr, X, y, X.columns.values, 'sepal length (cm)', 1)
print(shadow_forestry.get_children_left())
print(shadow_forestry.get_children_right())
# print(shadow_forestry.get_node_samples())
print(shadow_forestry.get_split_samples(6)) --- error for leaf node, indeces messed up
# print(shadow_forestry.get_split_samples(3)) --- Still wrong
# print(shadow_forestry.get_node_nsamples(6))

# print(shadow_forestry.get_score())


# Sklearn

regr = tree.DecisionTreeRegressor(max_depth=2)
regr.fit(X, y)

shadow_dtree = ShadowSKDTree(regr, X, y, X.columns.values, 'sepal length (cm)')
# print(shadow_dtree.get_children_left())
# print(shadow_dtree.get_children_right())
# print(shadow_dtree.get_features())
# dec_paths = regr.decision_path(X)
# print(dec_paths)

