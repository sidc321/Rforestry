import pandas as pd
from sklearn.datasets import load_iris


def get_data():
    data = load_iris()
    data_frame = pd.DataFrame(data["data"], columns=data["feature_names"])
    data_frame["target"] = data["target"]
    X = data_frame.loc[:, data_frame.columns != "target"]
    y = data_frame["target"]
    return X, y
