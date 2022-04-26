
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import warnings
import math
import os
from random import randrange
import sys
from forestry import forestry
import Py_preprocessing
from sklearn.ensemble import RandomForestClassifier

df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])


y = [0, 0, 1]

res = Py_preprocessing.training_data_checker(
    x = df2,
    y = y,
    ntree = 3,
    replace = True,
    sampsize = 4,
    mtry = 3,
    nodesizeSpl = 2,
    nodesizeAvg = 1,
    nodesizeStrictSpl = 2,
    nodesizeStrictAvg = 1,
    minSplitGain = 0,
    maxDepth = 10,
    interactionDepth = 8,
    splitratio = 0.5,
    OOBhonest = 0,
    nthread = 7,
    middleSplit = True,
    doubleTree = 0,
    linFeats = [0, 1, 2],
    monotonicConstraints = [0, 0, 0],
    groups = pd.Series([1,2,3], dtype="category"),
    featureWeights = 0,
    deepFeatureWeights = 0,
    observationWeights = [1, 2, 3],
    linear = False,
    symmetric = [0,1,0,1],
    scale = False,
    hasNas = False
)

#ONLY NUMERUCAL CATEGORICAL DATA?????

new_dat = pd.Categorical(['1', '1', 'o'])
df2['cat'] = new_dat

cat2 = pd.Series([1,3,5], dtype="category")
df2['cat2'] = cat2


# preproc_test = Py_preprocessing.preprocess_training(df2, y)
# print(preproc_test)

test_check = Py_preprocessing.preprocess_testing(df2, [], [])

a = {1, 2}
b = {1,2}
if a-b:
    print(a-b)
# PF1 = forestry(x=df2, y=y, groups=pd.Categorical([1,1,1,2,3,2,3,2,3,3,4,2,1,2,2,3,3,3,4]), minTreesPerGroup=100)


