
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
        ntree = 500,
        maxDepth=2,
        verbose=False,
        scale=False
)

print("Fitting the forest")
fr.fit(X, y)

#%%

print("Predicting with the forest")
forest_preds = fr.predict(aggregation='oob')
print(forest_preds)

#%%

f

#%%
print("Get the variable importance")
print(fr.getVI())

print("Get the OOB Error")
print(fr.getOOB())

#%%
# Now try getting a tree translated
fr.translate_tree_python()
print(fr.Py_forest)

#%%
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import platform
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from forestry_shadow import ShadowForestryTree

#%%

regr = tree.DecisionTreeRegressor(max_depth=3)
boston = load_boston()
regr.fit(boston.data, boston.target)

#print(regr.decision_path(boston.data))
#path_data = regr.decision_path(boston.data)
shadow_dtree = ShadowSKDTree(regr, boston.data, boston.target, boston.feature_names, "price")


#%%

viz = dtreeviz(shadow_dtree,
               boston.data,
               boston.target,
               target_name='price',
               feature_names=boston.feature_names,
               scale=2.0)
viz.view()


#%%
# See a classification example
clas = tree.DecisionTreeClassifier(max_depth=2)
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

viz = dtreeviz(clas,
               X_train,
               y_train,
               target_name='Species',
               scale = 3.0,
               feature_names=iris.feature_names,
               class_names=["setosa", "versicolor", "virginica"],
               histtype= 'barstacked')  # barstackes is default
viz.view()

#%%

clf = tree.DecisionTreeClassifier(max_depth=2)
wine = load_wine()

X_train = wine.data
y_train = wine.target
clf.fit(X_train, y_train)

# pick random X observation for demo
X = wine.data[np.random.randint(0, len(wine.data)),:]

viz = dtreeviz(clf,
               wine.data,
               wine.target,
               target_name='wine',
               scale = 2.0,
               feature_names=wine.feature_names,
               class_names=list(wine.target_names),
               X=X)  # pass the test observation
viz.view()