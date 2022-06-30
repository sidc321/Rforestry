
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


# data = load_iris()
# df = pd.DataFrame(data['data'], columns=data['feature_names'])
# df['target'] = data['target']
# X = df.loc[:, df.columns != 'target']
# X = X.loc[:, X.columns != 'sepal length (cm)']
# y = df['sepal length (cm)']


# fr = forestry(
#         ntree = 1,
#         maxDepth=2,
#         interactionDepth=2,
#         verbose=False,
#         scale=False,
#         seed=1729
# )

# fr.fit(X, y)

# print(fr.score(X, y))

# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Getting the dataset
# X, y = fetch_california_housing(return_X_y=True)

# # Splitting the data into testing and training datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Create a forestry object
# fr = forestry()

# print('Traingng the forest')
# fr.fit(X_train, y_train)

# print('Making predictions')
# preds = fr.predict(X_test)

# from sklearn.metrics import r2_score
# print(y_test, preds)
# print(r2_score(y_test, preds))

# print('The coefficient of determination is' + 
#         str(fr.score(X_test, y_test)))

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

# Getting the dataset
X, y = fetch_california_housing(return_X_y=True)

# Splitting the data into testing and training datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a forestry object
fr = forestry(scale=False)

print('Traingng the forest')
fr.fit(X_train, y_train)

print('Making predictions')
preds = fr.predict(X_test)

print('The coefficient of determination is' +
        str(fr.score(X_test, y_test)))