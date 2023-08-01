# pylint: disable=redefined-outer-name

import numpy as np
import pandas as pd
import pytest
from helpers import get_data
from sklearn.datasets import load_iris

from random_forestry import RandomForest


@pytest.fixture
def forest1():
    forest = RandomForest(seed=432432)

    X, y = get_data()

    forest.fit(X, y)
    return forest


@pytest.fixture
def forest2():
    X, y = get_data()

    forest = RandomForest(seed=432432, ntree=10, replace=False, sampsize=len(y), mtry=X.shape[1])
    forest.fit(X, y)
    return forest


def test_lambda0_predictions(forest1: RandomForest):
    X, _ = get_data()

    predictions_1 = forest1.predict(X, hier_shrinkage=True, lambda_shrinkage=0)
    predictions_2 = forest1.predict(X)
    assert np.array_equal(predictions_1, predictions_2)


def test_lambdalarge_predictions(forest2: RandomForest):
    X, y = get_data()

    predictions = forest2.predict(X, hier_shrinkage=True, lambda_shrinkage=1e13)
    tot_prediction_diffs = predictions - np.mean(y)
    assert np.allclose(tot_prediction_diffs, np.zeros(len(y)))


def test_small_tree_shrinkage():
    data = load_iris()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    X = df.loc[:, df.columns == "sepal width (cm)"]
    y = df["sepal length (cm)"]

    forest = RandomForest(seed=432432, ntree=1, max_depth=1, scale=False)
    forest.fit(X, y)
    forest.translate_tree()

    # classify training data
    fdata = forest.saved_forest[0]
    expectedPredictions = np.zeros(X.shape[0])
    for i in range(len(expectedPredictions)):
        if X.iloc[i, 0] < fdata["threshold"][0]:
            expectedPredictions[i] = 1
        else:
            expectedPredictions[i] = 2

    lambda_shrinkage = 2
    weightLeftPath = fdata["values"][0] * (1 - 1 / (1 + lambda_shrinkage / fdata["average_count"][0])) + fdata[
        "values"
    ][1] / (1 + lambda_shrinkage / fdata["average_count"][0])
    weightRightPath = fdata["values"][0] * (1 - 1 / (1 + lambda_shrinkage / fdata["average_count"][0])) + fdata[
        "values"
    ][2] / (1 + lambda_shrinkage / fdata["average_count"][0])
    expectedPredictions[expectedPredictions == 1] = weightLeftPath
    expectedPredictions[expectedPredictions == 2] = weightRightPath
    shrinked_pred = forest.predict(X, hier_shrinkage=True, lambda_shrinkage=lambda_shrinkage)
    assert np.array_equal(shrinked_pred, expectedPredictions)


def test_shrink_weight_matrix():
    forest = RandomForest(seed=432432)
    X, y = get_data()
    forest.fit(X, y)

    shrink_preds = forest.predict(X, return_weight_matrix=True, hier_shrinkage=True, lambda_shrinkage=10)
    # now we reconstruct predictions from the weight matrix and check they match
    weight_preds = shrink_preds["weightMatrix"].dot(y)
    assert np.allclose(weight_preds, shrink_preds["predictions"])
