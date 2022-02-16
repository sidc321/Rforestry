import ctypes
from unittest import result

_lib = ctypes.CDLL('./libsum.so')

#############
#DOESN'T KNOW THE PARAMETERS AND THE RETURN TYPE
# PROVIDE argtypes and restypes 
# DEFAULT return value is an int

#prints consolePrint
_lib.consolePrint()


#for addone
print(_lib.addOne(5))

#for someFunc
_lib.someFunc.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int))

def someFunc(numbers):
    global _lib
    num_numbers = len(numbers)
    array_type = ctypes.c_int * num_numbers

    res = _lib.someFunc(ctypes.c_int(num_numbers), array_type(*numbers))

    return res

print(someFunc([1,2,3,4,5]))