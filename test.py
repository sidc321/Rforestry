import ctypes

print("Loading Forestry DLL")
forestry  = ctypes.CDLL("src/libforestryCpp.dylib")
print(forestry)
print(forestry.add_one(1))

forestry.train_forest.argtypes = ctypes.c_int
forestry.train_forest.restype = ctypes.c_void_p
forestry.vector_get.restype = ctypes.c_double
forestry.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]

forest_trained = forestry.train_forest(3)
forestry.vector_get(forest_trained, 0)

