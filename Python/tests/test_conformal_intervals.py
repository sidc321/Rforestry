import numpy as np
from helpers import get_data
from sklearn.model_selection import train_test_split

from Rforestry import RandomForest


def test_conformal_intervals():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    forest = RandomForest(oob_honest=True, seed=3242)
    forest.fit(X_train, y_train)

    predictions = forest.get_ci(newdata=X_test, level=0.95, method="OOB-conformal")
    assert np.sum((y_test < predictions["CI.upper"]) & (y_test > predictions["CI.lower"])) > 0
