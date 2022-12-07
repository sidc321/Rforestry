from forestry import RandomForest
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


# Getting the dataset
data = load_iris()
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = data["target"]
cat_col = np.random.choice(["a", "b", "c"], size=len(X.index))
X["CategoricalVar"] = pd.Categorical(cat_col)

# Create a RandomForest object
fr = RandomForest(ntree=100, linear=True, maxDepth=5, overfitPenalty=0.001, nodesizeStrictSpl=10, seed=1)
fr.fit(X.iloc[:, 1:], X.iloc[:, 1], linFeats=[0, 1])

fr2 = RandomForest(
    ntree=100, linear=True, maxDepth=5, overfitPenalty=0.001, nodesizeStrictSpl=10, seed=1, doubleBootstrap=True
)
fr2.fit(X.iloc[:, 1:], X.iloc[:, 1], linFeats=[0, 1])

print("translate the first tree")

fr.translate_tree_python(0)
print(fr.Py_forest[0]["children_left"].size)
fr2.translate_tree_python(0)
print(fr2.Py_forest[0])

print("Making predictions")
preds = fr2.predict(X.iloc[:, 1:], aggregation="doubleOOB", weightMatrix=True)
print(preds)
fr2.save_forestry("rforest")

fr_load = fr2.load_forestry("rforest")
preds_after = fr_load.predict(weightMatrix=True, aggregation="doubleOOB")
print(preds["weightMatrix"])

print("\n The two predictions are equal: " + str(np.array_equal(preds["weightMatrix"], preds_after["weightMatrix"])))


for i in range(150):
    print(
        "\n The two predictions are equal: "
        + str(np.array_equal(preds["weightMatrix"][i], preds_after["weightMatrix"][i]))
    )
