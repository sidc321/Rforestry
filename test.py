import ctypes
import re
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import math
from random import randrange

print("Loading Forestry DLL")
forestry = (ctypes.CDLL("src/libforestryCpp.so"))  #CHANGE TO DLL IF NECESSARY
#forestry = (ctypes.CDLL("src/libforestryCpp.dylib")) 
print(forestry)

#Setting up argument types and result types 
forestry.get_data.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_bool,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int, 
    ctypes.c_int
    ]
forestry.get_data.restype =  ctypes.c_void_p

forestry.train_forest.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_bool,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_uint,
    ctypes.c_size_t,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_double,
    ctypes.c_bool
]
forestry.train_forest.restype = ctypes.c_void_p

forestry.predict_forest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                    ctypes.c_int, ctypes.c_bool]
forestry.predict_forest.restype =  ctypes.c_void_p

forestry.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
forestry.vector_get.restype = ctypes.c_double


# ref : https://stackoverflow.com/questions/58727931/how-to-pass-a-2d-array-from-python-to-c

def get_data_pointer(data):
    kdoublePtr = ctypes.POINTER(ctypes.c_double)
    kdoublePtrPtr = ctypes.POINTER(kdoublePtr)
    ct_arr = np.ctypeslib.as_ctypes(data)
    doublePtrArr = kdoublePtr * ct_arr._length_
    ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr)), kdoublePtrPtr)
    return ct_ptr

class Forestry:

    def __init__(
        self, 
        ntree = 5,
        replace = True,
        splitRatio = 1,
        OOBhonest = False,
        doubleBootstrap = False,
        mtry = 1,
        minNodeSizeSpt = 1,
        minNodeSizeAvg = 1,
        minNodeSizeToSplitSpt = 1,
        minNodeSizeToSplitAvg = 1,
        minSplitGain = 0,
        maxDepth = 10,
        interactionDepth = 100,
        seed = randrange(1001),
        nthread = 0,
        verbose = True,
        splitMiddle = True,
        maxObs = None,
        minTreesPerGroup = 0,
        hasNas = False,
        linear = False,
        symmetric = False,
        overfitPenalty = 1,
        doubleTree = False
    ):
        self.ntree = ntree
        self.replace = replace
        self.splitRatio = splitRatio
        self.OOBhonest = OOBhonest
        self.doubleBootstrap = doubleBootstrap
        self.mtry = mtry
        self.minNodeSizeSpt = minNodeSizeSpt
        self.minNodeSizeAvg = minNodeSizeAvg
        self.minNodeSizeToSplitSpt = minNodeSizeToSplitSpt
        self.minNodeSizeToSplitAvg = minNodeSizeToSplitAvg
        self.minSplitGain = minSplitGain
        self.maxDepth = maxDepth
        self.interactionDepth = interactionDepth
        self.seed = seed
        self.nthread = nthread
        self.verbose = verbose
        self.splitMiddle = splitMiddle
        self.maxObs = maxObs
        self.minTreesPerGroup = minTreesPerGroup
        self.hasNas = hasNas
        self.linear = linear
        self.symmetric = symmetric
        self.overfitPenalty = overfitPenalty
        self.doubleTree = doubleTree

        self.forest_pointer = None
        self.data_pointer = None
        self.sampSize = None

    def fit(self, X, y, categoricalFeatureCols, linearFeatures, featureWeights, featureWeightsVars, obsWeights, monotone_constr, groups, monotoneAvg, symmetricIndices, sampSize=None):
        X = pd.concat([X, y], axis=1)
        array = np.ascontiguousarray(X.values[:,:], np.double)
        ct_ptr = get_data_pointer(array)

        categoricalFeatureCols_arr = np.ascontiguousarray(categoricalFeatureCols, np.intc)
        categoricalFeatureCols_ptr = np.ctypeslib.as_ctypes(categoricalFeatureCols_arr)

        linearFeatures_arr = np.ascontiguousarray(linearFeatures, np.intc)
        linearFeatures_ptr = np.ctypeslib.as_ctypes(linearFeatures_arr)

        featureWeights_arr = np.ascontiguousarray(featureWeights, np.double)
        featureWeights_ptr = np.ctypeslib.as_ctypes(featureWeights_arr)

        featureWeightsVars_arr = np.ascontiguousarray(featureWeightsVars, np.intc)
        featureWeightsVars_ptr = np.ctypeslib.as_ctypes(featureWeightsVars_arr)

        obsWeights_arr = np.ascontiguousarray(obsWeights, np.double)
        obsWeights_ptr = np.ctypeslib.as_ctypes(obsWeights_arr)

        monotone_constr_arr = np.ascontiguousarray(monotone_constr, np.intc)
        monotone_constr_ptr = np.ctypeslib.as_ctypes(monotone_constr_arr)

        groups_arr = np.ascontiguousarray(groups, np.intc)
        groups_ptr = np.ctypeslib.as_ctypes(groups_arr)

        symmetricIndices_arr = np.ascontiguousarray(symmetricIndices, np.intc)
        symmetricIndices_ptr = np.ctypeslib.as_ctypes(symmetricIndices_arr)

        data_pr = ctypes.c_void_p(forestry.get_data(
            ct_ptr, 
            categoricalFeatureCols_ptr, categoricalFeatureCols_arr.size, 
            linearFeatures_ptr, linearFeatures_arr.size,
            featureWeights_ptr,
            featureWeightsVars_ptr, featureWeightsVars_arr.size,
            obsWeights_ptr,
            monotone_constr_ptr,
            groups_ptr,
            monotoneAvg,
            symmetricIndices_ptr, symmetricIndices_arr.size,
            array.shape[0], array.shape[1]
            ))



        #Set sampSize and maxObs fields
        if sampSize is None:
            self.sampSize = len(X.index) if self.replace else math.ceil(0.632 * len(X.index))
        else:
            self.sampSize = sampSize
        
        if self.maxObs is None:
            self.maxObs = len(X.index)

        forest_trained = ctypes.c_void_p(forestry.train_forest(
            data_pr, 
            self.ntree, 
            self.replace,
            self.sampSize,
            self.splitRatio,
            self.OOBhonest,
            self.doubleBootstrap,
            self.mtry,
            self.minNodeSizeSpt,
            self.minNodeSizeAvg,
            self.minNodeSizeToSplitSpt,
            self.minNodeSizeToSplitAvg,
            self.minSplitGain,
            self.maxDepth,
            self.interactionDepth,
            self.seed,
            self.nthread,
            self.verbose,
            self.splitMiddle,
            self.maxObs,
            self.minTreesPerGroup,
            self.hasNas,
            self.linear,
            self.symmetric,
            self.overfitPenalty,
            self.doubleTree
        ))

        self.data_pointer = data_pr
        self.forest_pointer = forest_trained

    
    def predict(self, X_test):

        if self.forest_pointer is None:
            raise ValueError("Cannot predict before training on a dataset.")

        X_test = pd.concat([X_test], axis=1)
        array_test = np.ascontiguousarray(X_test.values[:,:], np.double)
        ct_ptr_test = get_data_pointer(array_test)

        forest_preds = forestry.predict_forest(self.forest_pointer, 
                                               self.data_pointer, ct_ptr_test, 
                                               array_test.shape[0],
                                               self.verbose)

        res = np.empty(len(X_test.index))
        for i in range(len(X_test.index)):
            res[i] = forestry.vector_get(forest_preds, i)
        
        return res


# Load in pandas data
data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
df.head()
X = df.loc[:, df.columns != 'target']
y = df['target']

art_cat1 = np.concatenate((np.repeat(0, 30), np.repeat(1, 55), np.repeat(2, 15), np.repeat(3, 50)), axis=0)
art_cat2 = np.concatenate((np.repeat(0, 30), np.repeat(1, 40), np.repeat(2, 30), np.repeat(3, 50)), axis=0)
np.random.shuffle(art_cat1)
np.random.shuffle(art_cat2)
X = pd.concat([X, pd.Series(art_cat1)], axis=1)
X = pd.concat([X, pd.Series(art_cat2)], axis=1)

# !!!MINSPLITGAIN minTreesPerGroup BUG!!!
# Using push_back?


fr = Forestry(5, replace=True, splitRatio=0.5, OOBhonest=True, doubleBootstrap=True, mtry=4, minNodeSizeSpt=4, minNodeSizeToSplitSpt=4, maxDepth=20, seed=729, nthread=4, verbose=False, maxObs=160, linear=True, overfitPenalty=10, doubleTree=True)

print("Fitting the forest")
fr.fit(X, y, categoricalFeatureCols=np.array([4,5]), linearFeatures=np.arange(6), featureWeights=(np.array([0.1,0.2,0.2,0.3,0.1,0.1])), featureWeightsVars=[0,1,2,3,4], obsWeights=np.repeat(1/150, 150), monotone_constr=np.repeat(0,6), groups=np.repeat(0,150), monotoneAvg=False, symmetricIndices=[1])

X_test = df.loc[:, df.columns != 'target']

print("Predicting with the forest")
forest_preds = fr.predict(X_test)

print(forest_preds)
