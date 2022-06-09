
import numpy as np
import pandas as pd
import warnings
import math
import os
from random import randrange
import sys
from forestry import forestry
import Py_preprocessing
from sklearn.datasets import load_iris

#%%

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
X = X.loc[:, X.columns != 'sepal length (cm)']
y = df['sepal length (cm)']

fr = forestry(
        ntree = 1,
        maxDepth=2,
        verbose=True,
        scale=False
)

print("Fitting the forest")
fr.fit(X, y)

#print("Predicting with the forest")
#forest_preds = fr.predict(aggregation='oob')
#print(forest_preds)


#%%
# Now try getting a tree translated
fr.translate_tree_python()
print(fr.Py_forest)

#%%
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import platform
#from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from forestry_shadow import ShadowForestryTree

#%%

regr = tree.DecisionTreeRegressor(max_depth=2)
boston = load_boston()
regr.fit(boston.data, boston.target)

print(regr.decision_path(boston.data))

path_data = regr.decision_path(boston.data)

shadow_dtree = ShadowForestryTree(fr, boston.data, boston.target, boston.feature_names, "price")


#%%

viz = dtreeviz(shadow_dtree,
               boston.data,
               boston.target,
               target_name='price',
               feature_names=boston.feature_names,
               scale=3.0)
viz.view()


#%%
