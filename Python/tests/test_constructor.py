import pytest
from helpers import get_data

from random_forestry import RandomForest

X, y = get_data()


def test_properties():
    assert RandomForest(oob_honest=True).fit(X, y, splitratio=0.3).splitratio_ == 1

    assert RandomForest(oob_honest=True).fit(X, y, replace=False).replace_

    # assert RandomForest().fit(X, y, splitratio=0, double_tree=True).double_tree_ is False
    # assert RandomForest().fit(X, y, splitratio=0.3, double_tree=True).double_tree_
    assert RandomForest().fit(X, y, splitratio=1, double_tree=True).double_tree_ is False

    assert RandomForest().fit(X, y, interaction_depth=23, max_depth=4).interaction_depth_ == 4

    # with pytest.raises(ValidationError):
    #    RandomForest(ntree=False)

    # with pytest.raises(ValidationError):
    #    RandomForest(verbose=12)

    with pytest.raises(ValueError):
        RandomForest().fit(X, y, splitratio=1.4)

    # with pytest.raises(ValueError):
    #    RandomForest(min_split_gain=0.2, linear=False)


@pytest.mark.parametrize("test_input", [0, 3.6, None])
def test_ntree(test_input):
    with pytest.raises(ValueError, match=("^ntree must be positive integer$")):
        RandomForest(ntree=test_input).fit(X, y)


def test_sample_fraction():
    with pytest.raises(ValueError, match=("^sample_fraction must be positive or None$")):
        RandomForest(sample_fraction=-2).fit(X, y)
        # When sample_fraction=0, it results 'IndexError: vector' - needs to be investigated
        # RandomForest(sample_fraction=0).fit(X, y)


@pytest.mark.parametrize("test_input", [0, 3.6, None])
def test_nodesize_spl(test_input):
    with pytest.raises(ValueError, match=("^nodesize_spl must be positive integer$")):
        RandomForest(nodesize_spl=test_input).fit(X, y)


@pytest.mark.parametrize("test_input", [0, 3.6, None])
def test_nodesize_avg(test_input):
    with pytest.raises(ValueError, match=("^nodesize_avg must be positive integer$")):
        RandomForest(nodesize_avg=test_input).fit(X, y)
