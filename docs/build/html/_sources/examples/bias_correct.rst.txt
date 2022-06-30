Bias Corrected Predictions
===========================

This is an example how to use bias correction to make predictions.

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!
    from sklearn.datasets import load_breast_cancer
    import numpy as np

    # Getting the dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Create a forestry object
    fr = forestry(scale=False, OOBhonest=True)
    fr.fit(X, y)

    # Getting the bias corrected predictions
    corrected_preds = fr.correctedPredict(feats=[0,1,-1], nrounds=10, double=False,
            simple=False, params_forestry={'scale':False, 'OOBhonest':True})

    # Finding the out of bag error before and after
    print('OOB error before correction: ' + str(fr.getOOB()))
    print('OOB error after correction: ' + str(np.mean((corrected_preds - y)**2)))


