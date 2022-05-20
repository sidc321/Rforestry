import ctypes
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

print("Loading Forestry DLL")
forestry = (ctypes.CDLL("src/libforestryCpp.so"))  #CHANGE TO DLL IF NECESSARY
#forestry = (ctypes.CDLL("src/libforestryCpp.dylib")) 
print(forestry)

#Setting up argument types and result types 
forestry.get_data.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int]
forestry.get_data.restype =  ctypes.c_void_p

forestry.train_forest.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_bool]
forestry.train_forest.restype = ctypes.c_void_p

forestry.predict_forest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                    ctypes.c_int, ctypes.c_bool]
forestry.predict_forest.restype =  ctypes.c_void_p

forestry.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
forestry.vector_get.restype = ctypes.c_double


# ref : https://stackoverflow.com/questions/58727931/how-to-pass-a-2d-array-from-python-to-c

def get_data_pointer(array):
    kdoublePtr = ctypes.POINTER(ctypes.c_double)
    kdoublePtrPtr = ctypes.POINTER(kdoublePtr)
    ct_arr = np.ctypeslib.as_ctypes(array)
    doublePtrArr = kdoublePtr * ct_arr._length_
    ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr)), kdoublePtrPtr)
    return ct_ptr


class Forestry:

    def __init__(self, ntree, verbose=True):
        self.ntree = ntree
        self.forest_pointer = None
        self.data_pointer = None
        self.verbose = verbose

    def fit(self, X, y):
        X = pd.concat([X, y], axis=1)
        array = np.ascontiguousarray(X.values[:,:], np.double)
        ct_ptr = get_data_pointer(array)

        data_pr = ctypes.c_void_p(forestry.get_data(ct_ptr, array.shape[0], array.shape[1]))

        forest_trained = ctypes.c_void_p(forestry.train_forest(self.ntree, data_pr, self.verbose))

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


fr = Forestry(5)
print("Fitting the forest")
fr.fit(X, y)

X_test = df.loc[:, df.columns != 'target']
print("Predicting with the forest")
forest_preds = fr.predict(X_test)

print(forest_preds)
