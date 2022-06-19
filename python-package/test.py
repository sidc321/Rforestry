
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
#%%

# Load in the training data

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

#%%
fr.fit(X, y)
#%%
print("Predicting with the forest")
forest_preds = fr.predict(newdata = X)
print(forest_preds)
#%%
fr.translate_tree_python(tree_id=5)

#%%
print(forest_preds)
shadow_forestry = ShadowForestryTree(fr, X, y, X.columns.values, 'sepal length (cm)', 1)
fr.translate_tree_python(tree_id=5)
#%%
viz = dtreeviz(shadow_forestry,
                scale=3.0,
                target_name='sepal length (cm)',
                feature_names=X.columns.values)

viz.view()

#%%
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import platform


# # print(shadow_forestry.get_children_right())
# # print(X.columns.values)
# viz = dtreeviz(shadow_forestry,
#                scale=3.0,
#                target_name='sepal length (cm)',
#                feature_names=X.columns.values)

# viz.view()
# #%%
# # print(shadow_forestry.get_node_samples())
# # print(shadow_forestry.get_split_samples(3)) --- Still wrong
# # print(shadow_forestry.get_node_nsamples(6))

# # print(shadow_forestry.get_score())

