import os
from re import T
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris
from forestry import forestry

import numpy as np
import pandas as pd

import pytest

def get_data():
    data = load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    return X, y

@pytest.fixture
def get_forest():
    forest = forestry(
    ntree = 500,
    replace = True,
    sample_fraction=0.8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    splitrule = 'variance',
    splitratio = 1,
    nodesizeStrictAvg = 5,
    seed = 2
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest

def test_translate_tree(get_forest):
    fr = get_forest

    assert fr.Py_forest == {}

    fr.translate_tree_python(0)

    assert fr.Py_forest is not None #Py_forest will be filled after translation
    
    numNodes = fr.Py_forest['children_right'].size
    assert not any(fr.Py_forest[key].size != numNodes for key in fr.Py_forest.keys())


def test_all_trees(get_forest):
    fr = get_forest
    X, y = get_data()

    assert fr.Py_forest == {}

    for i in range(fr.ntree):
        fr.translate_tree_python(i)
        assert fr.Py_forest is not None

        numNodes = fr.Py_forest['children_right'].size
        assert not any(fr.Py_forest[key].size != numNodes for key in fr.Py_forest.keys())

        assert np.amax(fr.Py_forest['children_right']) <= numNodes - 1
        assert np.amin(fr.Py_forest['children_right']) < 0
        assert 0 not in fr.Py_forest['children_right']

        assert np.amax(fr.Py_forest['children_left']) <= numNodes - 1
        assert np.amin(fr.Py_forest['children_left']) < 0
        assert 0 not in fr.Py_forest['children_left']

        assert np.amax(fr.Py_forest['feature']) <= X.shape[1] - 1



    
    
