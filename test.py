import ctypes

print("Loading Forestry DLL")
forestry = (ctypes.CDLL("src/libforestryCpp.dylib"))
print(forestry)
print(forestry.add_one(1))

forestry.train_forest.argtypes = [ctypes.c_int]
forestry.train_forest.restype = ctypes.c_void_p
forestry.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
forestry.vector_get.restype = ctypes.c_double
forestry.predict_forest.argtypes = [ctypes.c_void_p]
forestry.predict_forest.restype =  ctypes.c_void_p

#ctypes.c_void_p


forest_trained = ctypes.c_void_p(forestry.train_forest(3))

forest_preds = forestry.predict_forest(forest_trained)
forest_preds
forestry.vector_get(forest_preds, 0)

