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

def test_translate_single_tree(get_forest):
    fr = get_forest
    assert len(fr.Py_forest) == fr.ntree

    fr.translate_tree_python(0)
    assert fr.Py_forest[0] #Py_forest[0] will be filled after translation
    assert all(fr.Py_forest[i] == dict() for i in range(1, fr.ntree))
    
    numNodes = fr.Py_forest[0]['children_right'].size
    assert not any(fr.Py_forest[0][key].size != numNodes for key in fr.Py_forest[0].keys())


def test_all_trees(get_forest):
    fr = get_forest
    X, y = get_data()
    assert len(fr.Py_forest) == fr.ntree

    fr.translate_tree_python(0)
    assert fr.Py_forest[0]

    # Translating more trees
    fr.translate_tree_python([0,1,2])
    assert fr.Py_forest[0]
    assert fr.Py_forest[1]
    assert fr.Py_forest[2]

    fr.translate_tree_python()

    for i in range(fr.ntree):
        assert fr.Py_forest[i]

        numNodes = fr.Py_forest[i]['children_right'].size
        assert not any(fr.Py_forest[i][key].size != numNodes for key in fr.Py_forest[i].keys())

        assert np.amax(fr.Py_forest[i]['children_right']) <= numNodes - 1
        assert np.amin(fr.Py_forest[i]['children_right']) < 0
        assert 0 not in fr.Py_forest[i]['children_right']

        assert np.amax(fr.Py_forest[i]['children_left']) <= numNodes - 1
        assert np.amin(fr.Py_forest[i]['children_left']) < 0
        assert 0 not in fr.Py_forest[i]['children_left']

        assert np.amax(fr.Py_forest[i]['feature']) <= X.shape[1] - 1



    
    
