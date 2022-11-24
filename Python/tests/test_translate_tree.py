import numpy as np
import pytest

from Rforestry import RandomForest
from helpers import get_data


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
    fr = forest
    assert not fr.Py_forest

    fr.translate_tree_python(0)
    assert len(fr.Py_forest) == fr.ntree
    assert fr.Py_forest[0]  # Py_forest[0] will be filled after translation
    assert all(fr.Py_forest[i] == dict() for i in range(1, fr.ntree))

    # numNodes = fr.Py_forest[0]['children_right'].size
    # assert not any(fr.Py_forest[0][key].size != numNodes for key in fr.Py_forest[0].keys() )


def test_all_trees(forest):
    fr = forest
    X, _ = get_data()

    fr.translate_tree_python(0)
    assert fr.Py_forest[0]
    assert len(fr.Py_forest) == fr.ntree

    # Translating more trees
    fr.translate_tree_python([0, 1, 2])
    assert fr.Py_forest[0]
    assert fr.Py_forest[1]
    assert fr.Py_forest[2]

    fr.translate_tree_python()

    for i in range(fr.ntree):
        assert fr.Py_forest[i]

        numNodes = fr.Py_forest[i]["children_right"].size
        # assert not any(fr.Py_forest[i][key].size != numNodes for key in fr.Py_forest[i].keys())

        assert np.amax(fr.Py_forest[i]["children_right"]) <= numNodes - 1
        assert np.amin(fr.Py_forest[i]["children_right"]) < 0
        assert 0 not in fr.Py_forest[i]["children_right"]

        assert np.amax(fr.Py_forest[i]["children_left"]) <= numNodes - 1
        assert np.amin(fr.Py_forest[i]["children_left"]) < 0
        assert 0 not in fr.Py_forest[i]["children_left"]

        assert np.amax(fr.Py_forest[i]["feature"]) <= X.shape[1] - 1
