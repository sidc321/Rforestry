from forestry import RandomForest
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Getting the dataset
data = load_iris()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']

# Create a RandomForest object
fr = RandomForest(OOBhonest=True, doubleBootstrap=True, scale=False)

print('Traingng the forest')
fr.fit(X, y)

print('Making doubleOOB predictions')
preds = fr.predict(aggregation='doubleOOB')
print(preds)