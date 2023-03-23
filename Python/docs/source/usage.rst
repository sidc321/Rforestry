Usage
======

Here are some examples of how to use the forestry package and some of its features. For a comprehensive
overview of all the classes and functions, check out the :doc:`API Reference <api>`.

.. contents:: Contents
    :depth: 2
    :local:


.. _set_get:

Setting the Parameters
-----------------------

Here is an example of how to use :meth:`get_params() <forestry.RandomForest.get_params>` 
and :meth:`set_params() <forestry.RandomForest.set_params>` to get and set the parameters 
of the forestry.

.. code-block:: Python

    from random_forestry import RandomForest

    # Create a RandomForest object
    fr = RandomForest(ntree=100, mtry=3, oob_honest=True)
    
    # Check out the list of parameters
    print(fr.get_parameters())

    # Modify some parameters
    newparams = {'max_depth': 10, 'oob_honest': False}
    fr.set_parameters(**newparams)
    fr.set_parameters(seed=1729)

    # Check out the new parameters
    print(fr.get_parameters())


.. _train_test:

Training and Testing
---------------------

Here is an example of how to train a forestry estimator and use it to make 
predictions. 

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Getting the dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Splitting the data into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Create a RandomForest object
    fr = RandomForest(scale=False)

    print('Traingng the forest')
    fr.fit(X_train, y_train)

    print('Making predictions')
    preds = fr.predict(X_test)

    print('The coefficient of determination is ' + 
            str(fr.score(X_test, y_test)))


.. _categorical:

Handling Categorical Data
--------------------------

Splits are made differently for categorical features. In order for the program to recognize that a given 
feature is categorical rather than continuous, the user must convert it into a
`Pandas categorical data type <https://pandas.pydata.org/docs/user_guide/categorical.html#>`_.

.. note::

    If a feature data is not numeric, the program will automatically consider it as a `Pandas categorical data type <https://pandas.pydata.org/docs/user_guide/categorical.html#>`_.

Here is an example of how to use categorical features.

.. code-block::

    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from random_forestry import RandomForest

    # Getting the dataset
    data = load_diabetes(as_frame=True, scaled=False).frame
    X = data.iloc[:, :-1]
    y = data['target']

    # Making 'sex' categorical
    X['sex'] = X['sex'].astype('category')

    # Splitting the data into testing and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Initialize a train
    fr = RandomForest()
    print('training the model')
    fr.fit(X_train, y_train)

    # Make predictions
    print('making predictions')
    preds = fr.predict(X_test)

    print('The coefficient of determination is ' + 
                str(fr.score(X_test, y_test)))


.. _oob:

Out of Bag Aggregation
-----------------------

This is an example of using out-of-bag aggregation. Check out :meth:`predict(..., aggregation='oob') <forestry.RandomForest.predict>` 
for more details.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a RandomForest object
    fr = RandomForest(oob_honest=True, scale=False)

    print('Traingng the forest')
    fr.fit(X, y)

    print('Making out-of-bag predictions')
    preds = fr.predict(aggregation='oob')
    print('OOB ERROR: ' + str(fr.get_oob()))


.. _doubleOOB:

Double Out of Bag Aggregation
-----------------------------

This is an example of using double OOB aggregation. Check out :meth:`predict(..., aggregation='doubleOOB') <forestry.RandomForest.predict>` 
for more details.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a RandomForest object
    fr = RandomForest(oob_honest=True, double_bootstrap=True, scale=False)

    print('Traingng the forest')
    fr.fit(X, y)

    print('Making doubleOOB predictions')
    preds = fr.predict(aggregation='doubleOOB')
    print(preds)

.. _vi:

Variable Importance
-------------------

This is an example how to get the variable importance. Check out the :meth:`API <forestry.RandomForest.get_vi>` 
for more details.

.. code-block:: Python

    from random_forestry import RandomForest
    from sklearn.datasets import load_breast_cancer
    import numpy as np

    # Getting the dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Create a RandomForest object and train
    fr = RandomForest(scale=False, max_depth=50)
    fr.fit(X, y)

    var_importance = fr.get_vi()
    print(var_importance)

.. _plot:

Plotting a Tree
----------------

To plot a specific tree in the forest, first convert it into a :class:`ShadowForestryTree <forestry_shadow.ShadowForestryTree>` object, 
then use the `dtreeviz <https://github.com/parrt/dtreeviz#usage>`_ library for visualization. Here is an example of how to do that.

.. code-block:: Python

    from random_forestry import RandomForest, ShadowForestryTree
    from dtreeviz.trees import *

    from sklearn.datasets import load_iris
    import numpy as np
    import pandas as pd

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a RandomForest object and train
    fr = RandomForest(ntree=100, max_depth=8)
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