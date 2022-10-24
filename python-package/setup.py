from setuptools import setup
import os
import sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, CURRENT_DIR)

setup(
    data_files=[("Lib/site-packages", ["libforestryCpp.dylib", "libforestryCpp.dylib"])]
)