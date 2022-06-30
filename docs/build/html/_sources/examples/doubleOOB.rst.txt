Double Out of Bag Aggregation
==============================

This is an example of using double out-of-bag aggregation.

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!
    from sklearn.datasets import load_iris
    import numpy as np

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = df['target']

    # Create a forestry object
    fr = forestry(OOBhonest=True, doubleBootstrap=True, scale=False)

    print('Traingng the forest')
    fr.fit(X, y)

    print('Making doubleOOB predictions')
    preds = fr.predict(aggregation='doubleOOB')
    print(preds)

