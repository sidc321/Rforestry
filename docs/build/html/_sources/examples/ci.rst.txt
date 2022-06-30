Confidence Intervals
=====================

This is an example how to get confidence intervals.

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!
    from sklearn.datasets import load_iris
    import numpy as np

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a forestry object
    fr = forestry(OOBhonest=True, doubleBootstrap=True, scale=False)
    fr.fit(X, y)

    conf_intervals = fr.getCI(newdata=X, method='OOB-conformal', level=.99)
    print(conf_intervals)

