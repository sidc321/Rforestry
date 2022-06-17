
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
X
y

#%%

fr = forestry(
    #ntree = 1,
    maxDepth=3,
    seed=1,
    verbose=True,
    scale=False
)


# print(fr.correctedPredict(X, nrounds=4, params_forestry={'ntree': 500, 'maxDepth': 2, 'verbose': False, 'scale': False}, feats=[0,1,2], simple=True, verbose=True, linear=True, keep_fits=True))
# print(fr.get_params())
#%%
print("Predicting with the forest")
forest_preds = fr.predict(newdata = X)
print(forest_preds)
#%%
print(forest_preds)
# shadow_forestry = ShadowForestryTree(fr, X, y, X.columns.values, 'sepal length (cm)', 1)
# # X.columns.values
# # #%%


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


# # Sklearn
# #%%
# regr = tree.DecisionTreeRegressor(max_depth=2)
# regr.fit(X, y)

# shadow_dtree = ShadowSKDTree(regr, X, y, X.columns.values, 'sepal length (cm)')

# #%%
# # print(shadow_dtree.get_children_left())
# # print(shadow_dtree.get_children_right())
# # print(shadow_dtree.get_features())
# # dec_paths = regr.decision_path(X)
# # print(dec_paths)


sk_fr = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=1729)
sk_fr.fit(X, y)

sk_fr.set_params(**{'n_estimators': 300, 'random_state': 1729})
print(sk_fr.get_params())