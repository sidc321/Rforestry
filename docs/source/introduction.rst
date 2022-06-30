Introduction
================


This is a basic overview of how to install and use the Python package. Check
out the :doc:`API Reference <api>` section for a comprehensive overview of all
the classses and functions, as well as the :doc:`Usage <examples/index>` section for a walkthrough of
how to use the main features of the package.

The forestry regressor uses the *Sklearn interface*. That means that a forestry object must be
:ref:`initialized <init>`, :ref:`trained <train>`, and then used to make :ref:`predictions <predict>`. A more detailed walkthrough can be found below.

.. contents:: Contents
    :depth: 2
    :local:


.. _install:

Installation
-------------

Rforestry is currently supported on Linux and MacOS. To install the python package,
simply use pip:

.. code-block:: console

   $ pip install Rforestry

To verify that the package is installed, try to import it using the following code:

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!

.. _init:

Initialization
---------------

To initialize a forestry object, simply use the class constructor:

.. code-block:: Python

   model = forestry()

To set the parameters, either pass them to the *forestry* constructor or use :meth:`set_params() <forestry.forestry.set_params>`. 
Note that the dataset must be passed during training and not initialization.


.. _train:

Training
---------------

To train the model, use the ``fit()`` method. ``fit()`` requires the feature martix and the target values to train the estimator.

.. code-block:: Python

   model = forestry()
   model.fit(X_train, y_train)

Check out the :doc:`API Reference <api>` for a more detailed overview of :meth:`fit() <forestry.forestry.fit>`.


.. _save-load:

Saving and Loading
-------------------

TO BE IMPLEMENTED


.. _predict: 

Predicting
---------------

To make predictions from a trained forest, use the ``predict()`` method.

.. code-block:: Python

   model = forestry()
   model.fit(X_train, y_train)

   preds = model.predict(X_test)

Check out the :doc:`API Reference <api>` for a more detailed overview of :meth:`predict() <forestry.forestry.predict>`.


Plotting
---------

TO BE IMPLEMENTED
