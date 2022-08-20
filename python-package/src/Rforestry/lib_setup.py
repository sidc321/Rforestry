import ctypes
import numpy as np
import pandas as pd


def setup_lib(lib):
    
    # set up the get_data function
    lib.get_data.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
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
        ctypes.c_uint
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


    # set up the vector_get function
    lib.vector_get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vector_get.restype = ctypes.c_double

    # Int get function
    lib.vector_get_int.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vector_get_int.restype = ctypes.c_int

    # Size_t get function
    lib.vector_get_size_t.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vector_get_size_t.restype = ctypes.c_size_t

    lib.get_prediction.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.get_prediction.restype = ctypes.c_double

    lib.get_weightMatrix.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.get_weightMatrix.restype = ctypes.c_double

    lib.getVI.argtypes = [ctypes.c_void_p]
    lib.getVI.restype = ctypes.c_void_p

    lib.get_tree_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.get_tree_info.restype = ctypes.c_void_p 

    lib.getTreeNodeCount.argtypes = [ctypes.c_void_p,ctypes.c_int]
    lib.getTreeNodeCount.restype = ctypes.c_int

    lib.get_path.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.get_path.restype = ctypes.c_void_p

    lib.py_reconstructree.argtypes = [
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
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint)
    ]
    lib.py_reconstructree.restype = ctypes.c_void_p


def get_data_pointer(data):
    data = np.ascontiguousarray(data.values[:,:], np.double)

    # kdoublePtr = ctypes.POINTER(ctypes.c_double)
    # kdoublePtrPtr = ctypes.POINTER(kdoublePtr)
    # ct_arr = np.ctypeslib.as_ctypes(data)
    # doublePtrArr = kdoublePtr * ct_arr._length_
    # ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, kdoublePtr) for row in ct_arr)), kdoublePtrPtr)
    # return ct_ptr

    # c_double_p = ctypes.POINTER(ctypes.c_double)
    # in_array_ptrs = (c_double_p * len(data))(*(r.ctypes.data_as(c_double_p) for r in data))
    # return in_array_ptrs

    return get_array_pointer(data.ravel(), np.double)



def get_array_pointer(array, dtype):
    return np.ctypeslib.as_ctypes(np.ascontiguousarray(array, dtype))
