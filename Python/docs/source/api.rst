API Reference
==============

Here, you can find the Python API reference of *Rforestry* classes.

.. contents:: Contents
    :depth: 3
    :local:

Random Forest Regressor
-----------------------
.. autoclass:: Rforestry.RandomForest
   :members:
   :undoc-members:

   .. rubric:: Methods

   .. autosummary::
         
      ~RandomForest.corrected_predict
      ~RandomForest.decision_path
      ~RandomForest.fit
      ~RandomForest.get_ci
      ~RandomForest.get_oob
      ~RandomForest.get_split_proportions
      ~RandomForest.get_vi
      ~RandomForest.get_parameters
      ~RandomForest.predict
      ~RandomForest.predict_info
      ~RandomForest.score
      ~RandomForest.set_parameters
      ~RandomForest.translate_tree


Plotting
---------

.. autoclass:: Rforestry.ShadowForestryTree
   :members:
   :undoc-members:

   .. rubric:: Methods

   .. autosummary::
         
      ~ShadowForestryTree.classes
      ~ShadowForestryTree.criterion
      ~ShadowForestryTree.get_children_left
      ~ShadowForestryTree.get_children_right
      ~ShadowForestryTree.get_class_weight
      ~ShadowForestryTree.get_class_weights
      ~ShadowForestryTree.get_feature_path_importance
      ~ShadowForestryTree.get_features
      ~ShadowForestryTree.get_max_depth
      ~ShadowForestryTree.get_min_samples_leaf
      ~ShadowForestryTree.get_node_criterion
      ~ShadowForestryTree.get_node_feature
      ~ShadowForestryTree.get_node_nsamples
      ~ShadowForestryTree.get_node_nsamples_by_class
      ~ShadowForestryTree.get_node_samples
      ~ShadowForestryTree.get_node_split
      ~ShadowForestryTree.get_prediction
      ~ShadowForestryTree.get_root_edge_labels
      ~ShadowForestryTree.get_score
      ~ShadowForestryTree.get_split_samples
      ~ShadowForestryTree.get_thresholds
      ~ShadowForestryTree.is_classifier
      ~ShadowForestryTree.is_fit
      ~ShadowForestryTree.nclasses
      ~ShadowForestryTree.nnodes
      ~ShadowForestryTree.shouldGoLeftAtSplit
