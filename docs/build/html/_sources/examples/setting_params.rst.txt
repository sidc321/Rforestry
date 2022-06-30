Setting the Parameters
=======================

Here is an example of how to use :meth:`get_params() <forestry.forestry.get_params>` 
and :meth:`set_params() <forestry.forestry.set_params>` to get and set the parameters 
of the forestry.

.. code-block:: Python

    from Rforestry.forestry import forestry ### Should be changed!!!!

    fr = forestry(ntree=100, mtry=3, OOBhonest=True)
    
    # Check out the list of parameters
    print(fr.get_params())

    # Modify some parameters
    newparams = {'maxDepth': 10, 'OOBhonest': False}
    fr.set_params(**newparams)
    fr.set_params(seed=1729)

    # Check out the new parameters
    print(fr.get_params())

