from setuptools import setup, Extension

setup(
    ext_modules=[Extension('mysum', ['mysum.cpp'],),],
)