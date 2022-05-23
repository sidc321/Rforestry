
from cmath import nan
from re import T
from tempfile import TemporaryDirectory
from traceback import print_tb
import numpy as np
import pandas as pd
import warnings
import math
import os
from random import randrange
import sys
from forestry import forestry
import Py_preprocessing
#from sklearn.ensemble import RandomForestClassifier

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

# #ONLY NUMERUCAL CATEGORICAL DATA?????

cat1 = pd.Series([1,1,3], dtype='category')
cat2 = pd.Series(['a', 'b', 'c'])

df2['cat1'] = cat1
df2['cat2'] = cat2


# fr = forestry(x=df2, y=y, scale=True, linear=True)


# co = df2.columns
# co = np.append(co, 'hey')
# print(co)

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
        OOBhonest = False,
        doubleBootstrap = None, #Add a default value.
        seed = randrange(1001),
        verbose = False,
        nthread = 0,
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

fr.fit(df2, y,  interactionVariables=[0], symmetric=[1,0,0,1,0])

print(fr.processed_dta)


# QUESTIONS

# 1) we set featureWeights[interactionVariables] = 0 only when featureWeights
#    is not provided. Shouldn't we always do this? How about for deepFeatureWeights? 
# 2) should we normalize featureWeights by dividing it by thee total sum?
# 3) sample_feature_checker feature_variables less than mtry
# 4) use push_back or not
# 5) symmetric indices or 0-1