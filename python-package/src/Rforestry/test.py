from forestry import RandomForest
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Getting the dataset
data = load_iris()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']
cat_col = np.random.choice(['a', 'b', 'c'], size=len(X.index))
X['CategoricalVar'] = cat_col

# Create a RandomForest object
fr = RandomForest(OOBhonest=True, doubleBootstrap=True, scale=False, ntree=500)

print('Traingng the forest')
fr.fit(X, y)

print('Making doubleOOB predictions')
preds = fr.predict(aggregation='doubleOOB')
print(preds)

fr.save_forestry('rforest.py')


fr_load = fr.load_forestry('rforest.py')
preds_after = fr_load.predict(aggregation='doubleOOB')
print(preds_after)

print('\n The two predictions are equal: ' + str(np.array_equal(preds, preds_after)))