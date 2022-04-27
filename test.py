import ctypes

print("Loading Forestry DLL")
forestry  = ctypes.CDLL("src/libforestryCpp.dylib")
print(forestry)
print(forestry.add_one(1))
