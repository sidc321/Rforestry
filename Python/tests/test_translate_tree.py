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
    assert not forest.forest

    forest.translate_tree(0)
    print("Length of pyforest")
    print(len(forest.forest))
    print(forest.forest[0]);

    assert len(forest.forest) == forest.ntree
    assert forest.forest[0]  # forest[0] will be filled after translation
    assert all(forest.forest[i] == {} for i in range(1, forest.ntree))

    # numNodes = fr.forest[0]['children_right'].size
    # assert not any(fr.forest[0][key].size != numNodes for key in fr.forest[0].keys() )


def test_all_trees(forest):
    X, _ = get_data()

    forest.translate_tree(0)
    assert forest.forest[0]
    assert len(forest.forest) == forest.ntree

    # Translating more trees
    forest.translate_tree([0, 1, 2])
    assert forest.forest[0]
    assert forest.forest[1]
    assert forest.forest[2]

    forest.translate_tree()

    for i in range(forest.ntree):
        assert forest.forest[i]

        num_nodes = forest.forest[i]["threshold"].size
        num_leaf_nodes = forest.forest[i]["values"].size
        # assert not any(forest.forest[i][key].size != numNodes for key in forest.forest[i].keys())
        assert len(forest.forest[i]["feature"]) == num_nodes+num_leaf_nodes
        assert len(forest.forest[i]["na_left_count"]) == num_nodes
        assert len(forest.forest[i]["na_right_count"]) == num_nodes
        assert len(forest.forest[i]["na_default_direction"]) == num_nodes

        assert np.amax(forest.forest[i]["splitting_sample_idx"]) <= X.shape[0]
        assert np.amax(forest.forest[i]["averaging_sample_idx"]) <= X.shape[0]

        assert np.amax(forest.forest[i]["feature"]) <= X.shape[1]
