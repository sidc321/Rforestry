from forestry import RandomForest
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Getting the dataset
data = load_iris()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']
cat_col = np.random.choice(['a', 'b', 'c'], size=len(X.index))
X['CategoricalVar'] = cat_col

# Create a RandomForest object
fr = RandomForest(ntree=500, OOBhonest=True)

print('Trainging the forest')
fr.fit(X, y, linFeats=[0, 1])

print('Making predictions')
preds = fr.predict(aggregation='oob')
print(preds)

# fr.save_forestry('rforest')


# fr_load = fr.load_forestry('rforest')
# preds_after = fr_load.predict(X)
# print(preds_after)

# print('\n The two predictions are equal: ' + str(np.array_equal(preds, preds_after)))


# fr = RandomForest()
# print(fr.test_array_passing(5))