
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

# Load in the training data
data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
X = X.loc[:, X.columns != 'sepal length (cm)']
y = df['sepal length (cm)']


#%%
X
y

#%%

fr = forestry(
    #ntree = 1,
    maxDepth=3,
    seed=1,
    verbose=False,
    scale=False
)

#%%
print("Fitting the forest")
fr.fit(X, y)

#%%
print("Predicting with the forest")
forest_preds = fr.predict(newdata = X)
print(forest_preds)
#%%
print(forest_preds)

print(fr.getVI())

#%%
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import platform


#%%

regr = tree.DecisionTreeRegressor(max_depth=2)
boston = load_boston()
regr.fit(boston.data, boston.target)
viz = dtreeviz(regr,
               boston.data,
               boston.target,
               target_name='price',
               feature_names=boston.feature_names)
viz.view()
