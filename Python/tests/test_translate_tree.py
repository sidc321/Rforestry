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
        nodesize_strict_spl=5,
        splitrule="variance",
        splitratio=1,
        nodesize_strict_avg=5,
        seed=2,
    )

    X, y = get_data()

    forest.fit(X, y)
    return forest


def test_translate_single_tree(forest):
    assert not forest.py_forest

    #forest.translate_tree(0)
    #assert len(forest.py_forest) == forest.ntree
    #assert forest.py_forest[0]  # py_forest[0] will be filled after translation
    #assert all(forest.py_forest[i] == {} for i in range(1, forest.ntree))

    # numNodes = fr.py_forest[0]['children_right'].size
    # assert not any(fr.py_forest[0][key].size != numNodes for key in fr.py_forest[0].keys() )


def test_all_trees(forest):
    X, _ = get_data()

