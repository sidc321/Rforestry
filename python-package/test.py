#%%

from forestry import RandomForest
from dtreeviz.trees import *
from forestry_shadow import ShadowForestryTree

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Getting the dataset
data = load_iris()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']

# Create a RandomForest object and train
fr = RandomForest(ntree=100, maxDepth=8)
fr.fit(X, y)

# Create a ShadowForestryTree object
shadow_forestry = ShadowForestryTree(fr, X, y, tree_id=28, feature_names=X.columns.values, target_name='Species')

# Plot the tree
viz = dtreeviz(shadow_forestry,
                scale=3.0,
                target_name='Species',
                feature_names=X.columns.values)

viz.view()


# Plot the prediction path of an observation
obs = X.loc[np.random.randint(0, len(X)),:]  # random sample from training

viz = dtreeviz(shadow_forestry, 
               target_name='Species', 
               orientation ='LR',  # left-right orientation
               feature_names=X.columns.values,
               X=obs)  # need to give single observation for prediction
              
viz.view()  

# See the prediction path in plain english
print(explain_prediction_path(shadow_forestry, x=obs, feature_names=X.columns.values, explanation_type='plain_english'))