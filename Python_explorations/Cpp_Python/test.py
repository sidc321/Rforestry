import numpy.ctypeslib as ctl
import ctypes

libname = 'testlib.so'
libdir = './'
lib=ctl.load_library(libname, libdir)

py_add_one = lib.add_one
py_add_one.argtypes = [ctypes.c_int]
value = 5
results = py_add_one(value)
print(results)