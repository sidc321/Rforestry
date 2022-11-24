from sklearn.datasets import load_iris
import pandas as pd


def get_data():
    data = load_iris()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    X = df.loc[:, df.columns != "target"]
    y = df["target"]
    return X, y
