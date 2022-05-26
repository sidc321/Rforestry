import ctypes
import numpy as np
import pandas as pd


def setup_lib(lib):
    
    # set up the get_data function
    lib.get_data.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t, 
        ctypes.c_size_t,
        ctypes.c_int
        ]
    lib.get_data.restype =  ctypes.c_void_p


    # set up the train_forest function
    lib.train_forest.argtypes = [
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
    lib.train_forest.restype = ctypes.c_void_p


    # set up the predict_forest function
    lib.predict_forest.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                        ctypes.c_int, ctypes.c_bool]
    lib.predict_forest.restype =  ctypes.c_void_p


    # set up the vector_get function
    lib.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vector_get.restype = ctypes.c_double


def get_data_pointer(data):
    data = np.ascontiguousarray(data.values[:,:], np.double)

    kdoublePtr = ctypes.POINTER(ctypes.c_double)
    kdoublePtrPtr = ctypes.POINTER(kdoublePtr)
    ct_arr = np.ctypeslib.as_ctypes(data)
    doublePtrArr = kdoublePtr * ct_arr._length_
    ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr)), kdoublePtrPtr)
    return ct_ptr


def get_array_pointer(array, dtype):
    arr = np.ascontiguousarray(array, dtype)
    return np.ctypeslib.as_ctypes(arr)
