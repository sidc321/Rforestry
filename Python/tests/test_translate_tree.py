# pylint: disable=redefined-outer-name

import numpy as np
import pytest
from helpers import get_data

from Rforestry import RandomForest


@pytest.fixture
def forest():
    forest = RandomForest(
        ntree=500,
        replace=True,
        sample_fraction=0.8,
        mtry=3,
        nodesizeStrictSpl=5,
        splitrule="variance",
        splitratio=1,
        nodesizeStrictAvg=5,
        seed=2,
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_translate_single_tree(forest):
    assert not forest.Py_forest

    forest.translate_tree_python(0)
    assert len(forest.Py_forest) == forest.ntree
    assert forest.Py_forest[0]  # Py_forest[0] will be filled after translation
    assert all(forest.Py_forest[i] == dict() for i in range(1, forest.ntree))

    # numNodes = fr.Py_forest[0]['children_right'].size
    # assert not any(fr.Py_forest[0][key].size != numNodes for key in fr.Py_forest[0].keys() )


def test_all_trees(forest):
    X, _ = get_data()

    forest.translate_tree_python(0)
    assert forest.Py_forest[0]
    assert len(forest.Py_forest) == forest.ntree

    # Translating more trees
    forest.translate_tree_python([0, 1, 2])
    assert forest.Py_forest[0]
    assert forest.Py_forest[1]
    assert forest.Py_forest[2]

    forest.translate_tree_python()

    for i in range(forest.ntree):
        assert forest.Py_forest[i]

        num_nodes = forest.Py_forest[i]["children_right"].size
        # assert not any(forest.Py_forest[i][key].size != numNodes for key in forest.Py_forest[i].keys())

        assert np.amax(forest.Py_forest[i]["children_right"]) <= num_nodes - 1
        assert np.amin(forest.Py_forest[i]["children_right"]) < 0
        assert 0 not in forest.Py_forest[i]["children_right"]

        assert np.amax(forest.Py_forest[i]["children_left"]) <= num_nodes - 1
        assert np.amin(forest.Py_forest[i]["children_left"]) < 0
        assert 0 not in forest.Py_forest[i]["children_left"]

        assert np.amax(forest.Py_forest[i]["feature"]) <= X.shape[1] - 1
