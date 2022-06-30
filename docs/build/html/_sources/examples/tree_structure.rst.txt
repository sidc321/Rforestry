Retrieve the Tree Structure
===========================

This is an example of how to retrieve the underlying tree structure in the forest. To do that,
we need to use the :meth:`translate_tree_python() <forestry.forestry.translate_tree_python>` function,
which fills the ``Py_forest`` attribute for the corresponding tree.

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!
    from sklearn.datasets import load_iris
    import numpy as np

    # Getting the dataset
    data = load_iris()
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = data['target']

    # Create a forestry object and train
    fr = forestry(scale=False, maxDepth=50)
    fr.fit(X, y)

    # Translate the first tree in the forest
    fr.translate_tree_python(0)
    print(fr.Py_forest[0])

    # Calculate the proportion of splits for each feature_names
    split_prop = fr.getSplitProps()
    print(split_prop)