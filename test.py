import ctypes
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# %%

print("Loading Forestry DLL")
forestry = (ctypes.CDLL("src/libforestryCpp.dylib"))
print(forestry)
print(forestry.add_one(1))


# Setting argument types for train_forest + vector_get functions
forestry.train_forest.argtypes = [ctypes.c_int, ctypes.c_void_p]
forestry.train_forest.restype = ctypes.c_void_p
forestry.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
forestry.vector_get.restype = ctypes.c_double

#ctypes.c_void_p

# Load in pandas data
data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
df.head()
X = df.loc[:, df.columns != 'target']
y = df['target']
X = pd.concat([X, y], axis=1)

# Now pass some data in
# X = pd.concat([X], axis=1)
# 
# ref : https://stackoverflow.com/questions/58727931/how-to-pass-a-2d-array-from-python-to-c
# create double pointer of type double (cast for 2d array input)

# we need to have a numpy contiguous array to pass into the C++ fit function

# Begin >FIT FUnction ==========================================================
array = np.ascontiguousarray(X.values[:,:], np.double)

kdoublePtr = ctypes.POINTER(ctypes.c_double)
kdoublePtrPtr = ctypes.POINTER(kdoublePtr)
ct_arr = np.ctypeslib.as_ctypes(array)
doublePtrArr = kdoublePtr * ct_arr._length_
ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr)), kdoublePtrPtr)

# Setting the argument types and return types for get_data
forestry.get_data.argtypes = [kdoublePtrPtr, ctypes.c_int, ctypes.c_int]
forestry.get_data.restype =  ctypes.c_void_p


# Get a pointer to the data frame
data_pr = ctypes.c_void_p(forestry.get_data(ct_ptr, array.shape[0], array.shape[1]))

# Train a forest and get a pointer to the forest
forest_trained = ctypes.c_void_p(forestry.train_forest(3, data_pr))
# End >FIT FUnction ============================================================

# Now predict on new data
X_test = df.loc[:, df.columns != 'target']
X_test = pd.concat([X_test], axis=1)
array_test = np.ascontiguousarray(X_test.values[:,:], np.double)

# Create types for prediction function arguments
ct_arr_test = np.ctypeslib.as_ctypes(array_test)
doublePtrArr = kdoublePtr * ct_arr_test._length_
ct_ptr_test = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr_test)), kdoublePtrPtr)

# Setting the arguments and returns for the predict function
forestry.predict_forest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, kdoublePtrPtr, ctypes.c_int]
forestry.predict_forest.restype =  ctypes.c_void_p

# This is actually doing predict, returns a pointer to a vector of doubles
forest_preds = forestry.predict_forest(forest_trained, data_pr, ct_ptr_test, array_test.shape[0])

forest_preds

# See what the actual predictions are by using the vector_get function
# First argument is a pointer, returned by predict_forest, and second is an index
print(forestry.vector_get(forest_preds, 0))
print(forestry.vector_get(forest_preds, 4))
print(forestry.vector_get(forest_preds, 9))
print(forestry.vector_get(forest_preds, 149))

