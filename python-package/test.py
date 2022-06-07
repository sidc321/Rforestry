
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

df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 7, 8], [7, 8, 100]]),
                   columns=['a', 'b', 'c'])


y = [0, 0, 1]

# res = Py_preprocessing.training_data_checker(
#     x = df2,
#     y = y,
#     ntree = 3,
#     replace = True,
#     sampsize = 4,
#     mtry = 3,
#     nodesizeSpl = 2,
#     nodesizeAvg = 1,
#     nodesizeStrictSpl = 2,
#     nodesizeStrictAvg = 1,
#     minSplitGain = 0,
#     maxDepth = 10,
#     interactionDepth = 8,
#     splitratio = 0.5,
#     OOBhonest = 0,
#     nthread = 7,
#     middleSplit = True,
#     doubleTree = 0,
#     linFeats = [0, 1, 2],
#     monotonicConstraints = [0, 0, 0],
#     groups = pd.Series([1,2,3], dtype="category"),
#     featureWeights = 0,
#     deepFeatureWeights = 0,
#     observationWeights = [1, 2, 3],
#     linear = False,
#     symmetric = [0,1,0,1],
#     scale = False,
#     hasNas = False
# )

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
y = df['target']

art_cat1 = np.concatenate((np.repeat(0, 30), np.repeat(1, 55), np.repeat(2, 15), np.repeat(3, 50)), axis=0)
art_cat2 = np.concatenate((np.repeat(0, 30), np.repeat(1, 40), np.repeat(2, 30), np.repeat(3, 50)), axis=0)
np.random.shuffle(art_cat1)
np.random.shuffle(art_cat2)

art_df = pd.DataFrame({'cat1': art_cat2, 'cat2': art_cat2})
X = pd.concat([X, art_df], axis=1)

#%%

fr = forestry(
        ntree = 500,
        replace = True,
        sampsize = 3,  #Add a default value.
        sample_fraction = 0.7,
        mtry = None,    #Add a default value.
        nodesizeSpl = 5,
        nodesizeAvg = 5,
        nodesizeStrictSpl = 1,
        nodesizeStrictAvg = 1,
        minSplitGain = 0,
        maxDepth = None,  #Add a default value.
        interactionDepth = None,   #Add a default value.
        splitratio = 1,
        OOBhonest = True,
        doubleBootstrap = True, #Add a default value.
        seed = 12,
        verbose = True,
        nthread = 8,
        splitrule = 'variance',
        middleSplit = False,
        maxObs = None,    #Add a default value.
        linear = False,
        minTreesPerGroup = 0,
        monotoneAvg = False,
        overfitPenalty = 1,
        scale = True,
        doubleTree = False,
        reuseforestry = None,
        savable = True,
        saveable = True
)

#%%
print("Fitting the forest")
fr.fit(X, y)

#%%
print("Predicting with the forest")
forest_preds = fr.predict(aggregation='oob')
print(forest_preds)
#%%
print(forest_preds)

print(fr.getVI())

print(fr.getCI(X, method='OOB-conformal'))

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
