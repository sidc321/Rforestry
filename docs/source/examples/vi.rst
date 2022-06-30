Variable Importance
===================

This is an example how to get the variable importance.

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!
    from sklearn.datasets import load_breast_cancer
    import numpy as np

    # Getting the dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Create a forestry object and train
    fr = forestry(scale=False, maxDepth=50)
    fr.fit(X, y)

    var_importance = fr.getVI()
    print(var_importance)


    # VI DOESN'T WORK BECAUSE OF WEIGHTMATRIX 