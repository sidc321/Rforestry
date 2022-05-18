import ctypes
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# %%

print("Loading Forestry DLL")
forestry = (ctypes.CDLL("src/libforestryCpp.dylib"))
print(forestry)
print(forestry.add_one(1))

forestry.train_forest.argtypes = [ctypes.c_int,ctypes.c_void_p]
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
array = np.ascontiguousarray(X.values[:,:], np.double)


kdoublePtr = ctypes.POINTER(ctypes.c_double)
kdoublePtrPtr = ctypes.POINTER(kdoublePtr)
ct_arr = np.ctypeslib.as_ctypes(array)
doublePtrArr = kdoublePtr * ct_arr._length_
ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr)), kdoublePtrPtr)

forestry.get_data.argtypes = [kdoublePtrPtr, ctypes.c_int, ctypes.c_int]
forestry.get_data.restype =  ctypes.c_void_p


# Now train the forest
data_pr = ctypes.c_void_p(forestry.get_data(ct_ptr, array.shape[0], array.shape[1]))
forest_trained = ctypes.c_void_p(forestry.train_forest(3, data_pr))


# Now predict on new data
X_test = df.loc[:, df.columns != 'target']
X_test = pd.concat([X_test], axis=1)
array_test = np.ascontiguousarray(X_test.values[:,:], np.double)

ct_arr_test = np.ctypeslib.as_ctypes(array_test)
doublePtrArr = kdoublePtr * ct_arr_test._length_
ct_ptr_test = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr_test)), kdoublePtrPtr)

forestry.predict_forest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, kdoublePtrPtr, ctypes.c_int]
forestry.predict_forest.restype =  ctypes.c_void_p

forest_preds = forestry.predict_forest(forest_trained, data_pr, ct_ptr_test, array_test.shape[0])
forest_preds
print(forestry.vector_get(forest_preds, 0))
print(forestry.vector_get(forest_preds, 4))
print(forestry.vector_get(forest_preds, 9))

