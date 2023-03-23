Introduction
================


This is a basic overview of how to install and use the Python package. Check
out the :doc:`API Reference <api>` section for a comprehensive overview of all
the classses and functions, as well as the :doc:`Usage <usage>` section for a walkthrough of
how to use the main features of the package.

The forestry regressor uses the *Sklearn interface*. This means that a *RandomForest* object must be
:ref:`initialized <init>`, :ref:`trained <train>`, and then used to make :ref:`predictions <predict>`. 
A more detailed walkthrough can be found below.

.. contents:: Contents
    :depth: 2
    :local:


.. _install:

Installation
-------------

random_forestry is currently supported on Linux and MacOS. To install the python package,
simply use pip:

.. code-block:: console

   $ pip install random_forestry

To verify that the package is installed, try to import the *RandomForest* class using the following code:

.. code-block:: Python

    from random_forestry import RandomForest

.. _init:

Initialization
---------------

To initialize a RandomForest object, simply use the class constructor:

.. code-block:: Python

   model = RandomForest()

To set the parameters, either pass them to the *RandomForest* constructor or use :meth:`set_params() <forestry.RandomForest.set_params>`. 
Note that the dataset must be passed during training and not initialization.


.. _train:

Training
---------------

To train the model, use the :meth:`fit() <forestry.RandomForest.fit>` method. :meth:`fit() <forestry.RandomForest.fit>` requires the feature martix and the target values to train the estimator.

.. code-block:: Python

   model = RandomForest()
   model.fit(X_train, y_train)

Check out the :doc:`API Reference <api>` for a more detailed overview of :meth:`fit() <forestry.RandomForest.fit>`.


.. _save-load:

Saving and Loading
-------------------

TO BE IMPLEMENTED


.. _predict: 

Predicting
---------------

To make predictions from a trained forest, use the :meth:`predict() <forestry.RandomForest.predict>` method.

.. code-block:: Python

   model = RandomForest()
   model.fit(X_train, y_train)

   preds = model.predict(X_test)

Check out the :doc:`API Reference <api>` for a more detailed overview of :meth:`predict() <forestry.RandomForest.predict>`.


.. _plotting: 

Plotting
---------

For visualizing the trees, *random_forestry* uses the `dtreeviz <https://github.com/parrt/dtreeviz#readme>`_ python library. This library
provides a number of plots for visualizing regression trees. Check out the `Installation Guide <https://github.com/parrt/dtreeviz#install>`_ 
and the `Usage <https://github.com/parrt/dtreeviz#usage>`_ for further details.

In order to plot a given tree in the forest, it must first be converted into a :class:`ShadowForestryTree <forestry_shadow.ShadowForestryTree>` 
object, which can be passed to *dtreeviz*. Here is a simple example.

.. code-block:: Python

   from dtreeviz.trees import *
   from random_forestry import ShadowForestryTree

   # Create a ShadowForestryTree object. Since this is a single tree, tree_id must be specified (default=0)
   shadow_forestry = ShadowForestryTree(model, X, y, X.columns.values, 'tagret name', tree_id=0)

   viz = dtreeviz(shadow_forestry,
                scale=3.0,
                target_name='tagret name',
                feature_names=X.columns.values)

   viz.view()

Check out the :ref:`Usage <plot>` section for a detailed example.