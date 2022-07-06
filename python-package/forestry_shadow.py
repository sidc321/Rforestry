from collections import defaultdict
from typing import List, Mapping
import warnings
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight

from dtreeviz.models.shadow_decision_tree import ShadowDecTree


class ShadowForestryTree(ShadowDecTree):
    """
    The class implementing *ShadowDecTree* to enable easy plotting. Note that to plot a specific tree, one must
    initialize an instance of this class and pass it *dtreeviz*. Check out the :ref:`Usage <plot>` section for an example
    of how to do this.

    :param tree_model: The model to be visualized.
    :type tree_model: *RandomForest*
    :param x_data: The feature matrix.
    :type x_data: *pandas.DataFrame, numpy.ndarray*
    :param y_data: The target values.
    :type y_data: *pandas.Series, numpy.ndarray*
    :param feature_names: Features' names.
    :type feature_names: *array_like of shape [ncols,], optional, default=None*
    :param target_name: The target's name.
    :type target_name: *str, optional, default=None*
    :param tree_id: The id of the tree to be visualized (must be in *[0, tree_model.ntree)*).
    :type tree_id: *int*
    :param class_names: Class' names (in case of a classifier).
    :type class_names: *array_like, dict, optional, default=None*
    """
    def __init__(self, 
                 tree_model,
                 x_data: (pd.DataFrame, np.ndarray),
                 y_data: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 tree_id = 0,
                 class_names: (List[str], Mapping[int, str]) = None):

        if feature_names is not None:
            if len(feature_names) != x_data.shape[1]:
                raise ValueError('x_data and feature_names have different number of features')

        if isinstance(x_data, pd.DataFrame):
            ft_names = x_data.columns.values
            if feature_names is None:
                feature_names = ft_names
            else:
                if not np.array_equal(ft_names, feature_names):
                    warnings.warn('The feature names provided are different from the column names of x_data')

        self.tree_id = tree_id
        tree_model.translate_tree_python(tree_id)
        super().__init__(tree_model, x_data, y_data, feature_names, target_name, class_names)

    def is_fit(self):
        """
        Checks if the model has been trained.

        :return: *True* if the model has been trained.
        :rtype: *bool*

        """
        return getattr(self.tree_model, 'forest') is not None

    def is_classifier(self):
        """
        Checks if the model does classification or regression.

        :return: Whether the model does classification.
        :rtype: bool

        """
        ## TODO: Change if we implement classification as a different method
        return False

    def get_class_weights(self):
        """
        Gets the class weights of the tree model.

        :return: An array whose *i-th* element is the weight for the *i-th* class.
        :rtype: numpy.array of shape [n_classes,]
        """
        if self.is_classifier():
            unique_target_values = np.unique(self.y_data)
            return compute_class_weight(self.tree_model.class_weight, classes=unique_target_values, y=self.y_data)
 
    def get_thresholds(self):
        """
        Gets the threshold values for each node in the tree.

        :return: An array whose *i-th* element gives the splitting point (threshold) of the split
         in the node with id *i*. If leaf node, the *i-th* element of the array is *0.0*.
        :rtype: numpy.array of shape[n_nodes,]
        """
        return self.tree_model.Py_forest[self.tree_id]["threshold"]

    def get_features(self):
        """
        Gets the splitting features for each node in the tree.

        :return: An array whose *i-th* element gives the splitting feature of the split in the node with id *i*.
         If leaf node, the *i-th* element of the array is the negative number of observations in the averaging
         set of that node.
        :rtype: numpy.array of shape[n_nodes,]
        """
        return self.tree_model.Py_forest[self.tree_id]["feature"]

    def criterion(self):
        """
        Gets the function to measure the quality of a split. 
        Ex. Gini, entropy, MSE, MAE.

        :return: the criterion used to measure the quality of a split.
        :rtype: str
        """
        return 'SQUARED_ERROR'

    def get_class_weight(self):
        """
        Gets the class weights of the tree model. To be compared with :meth:`get_class_weights() <ShadowForestryTree.get_class_weights>`.

        :return: An array whose *i-th* element is the weight for the *i-th* class.
        :rtype: numpy.array of shape [n_classes,]
        """
        if self.is_classifier():
            return self.tree_model.class_weight

    def nclasses(self):
        """
        Gets the number of classes. If the tree does regression, returns 1.

        :return: The number of classes.
        :rtype: int
        """
        pass
        return 1

    def classes(self):
        """
        Gets the classes of the tree in case of classification.

        :return: Classes of the classification tree.
        :rtype: numpy.array of shape [n_classes,]
        """
        if self.is_classifier():
            return self.tree_model.classes_

    def get_node_samples(self):
        """
        Maps the node id to the sample indices considered at that node for splitting.

        :return: A dictionary mapping the id of the nodes to a list of the sample indices considered at that node.
        :rtype: dict
        """

        dec_paths = self.tree_model.decision_path(self.x_data, self.tree_id)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            for node_id in dec:
                node_to_samples[node_id].append(sample_i)

        return node_to_samples

    def get_split_samples(self, id):
        """
        Gets the left and right split indices from a node.

        :param id: The id of the node.
        :type id: *int*
        :return: A tuple of two arrays. The first one consists of the sample indices in the left side of the split,
         and the second one consists of the sample indices in the right side of the split,
        :rtype: tuple(numpy.array, numpy.array)
        """
        samples = np.array(self.get_node_samples()[id])
        node_X_data = self.x_data[samples, self.get_node_feature(id)]
        split = self.get_node_split(id)

        left = np.nonzero(node_X_data <= split)[0]
        right = np.nonzero(node_X_data > split)[0]

        return left, right

    def get_root_edge_labels(self):
        """
        Gets the labels for roots and edges.

        :return: A list denoting how to label the roots and the edges.
        :rtype: list
        """
        return ["&le;", "&gt;"]

    def get_node_nsamples(self, id):
        """
        Gets the number of samples for a given node.

        :param id: The id of the node.
        :type id: *int*
        :return: The number of samples considered in the node with a given id.
        :rtype: int
        """
        return len(self.get_node_samples()[id])

    def get_children_left(self):
        """
        For each node in the tree, gets the node id of its left child.

        :return: An array whose *i-th* element gives the node id of the left child of the node with id *i*.
         If leaf node, the *i-th* element of the array is *-1*.
        :rtype: numpy.array of shape[n_nodes,]
        """
        return self.tree_model.Py_forest[self.tree_id]["children_left"]

    def get_children_right(self):
        """
        For each node in the tree, gets the node id of its right child.

        :return: An array whose *i-th* element gives the node id of the right child of the node with id *i*.
         If leaf node, the *i-th* element of the array is *-1*.
        :rtype: numpy.array of shape[n_nodes,]
        """
        return self.tree_model.Py_forest[self.tree_id]["children_right"]

    def get_node_split(self, id) -> (int, float):
        """
        Gets the split value (threshold) of the node with a given id.

        :param id: The id of the node.
        :type id: *int*
        :return: The split value (threshold) of the split in the given node. If leaf node, returns *0.0*.
        :rtype: int, float
        """
        return self.tree_model.Py_forest[self.tree_id]["threshold"][id]

    def get_node_feature(self, id) -> int:
        """
        Gets the index of the split feature for a given node.

        :param id: The id of the node.
        :type id: *int*
        :return: The index of the feature on which the split is made in the given node. If leaf node, returns
         the negative number of observations in the averaging set of that node.
        :rtype: int
        """
        return self.tree_model.Py_forest[self.tree_id]["feature"][id]

    def get_node_nsamples_by_class(self, id):
        """
        In case of a classification tree, gets the number of samples in each class in a given node.

        :param id: The id of the node.
        :type id: *int*
        :return: An array whose *i-th* element is the number of samples in the *i-th* class in the given node.
        :rtype: numpy.array of shape[n_classes,]
        """
        if self.is_classifier():
            return self.tree_model.Py_forest[self.tree_id]["values"][id][0]

    def get_prediction(self, id):
        """
        Gets the prediction made by a given node.

        :param id: The id of the node.
        :type id: *int*
        :return: The prediction made by the given node. If not a leaf node, returns *0.0*.
        :rtype: int, float
        """
        if self.is_classifier():
            counts = self.tree_model.Py_forest[self.tree_id]["values"][id][0]
            return np.argmax(counts)
        else:
            return self.tree_model.Py_forest[self.tree_id]["values"][id]

    def nnodes(self):
        """
        Gets the number of nodes in the tree.

        :return: The number of nodes in the tree (both internal and leaf).
        :rtype: int
        """
        return len(self.tree_model.Py_forest[self.tree_id]["feature"])

    def get_node_criterion(self, id):
        """
        Gets the impurity at a given node.

        :param id: The id of the node.
        :type id: *int*
        :return: The impurity at the given node.
        :rtype: int, float
        """
        return self.tree_model.tree_.impurity[id]

    def get_feature_path_importance(self, node_list):
        """
        For a given list of nodes, returns the feature importance.

        :param node_list: A list of node ids.
        :type node_list: *array_like*
        :return: The feature importance for the given nodes.
        :rtype: numpy.array of shape[ncols,]
        """
        gini = np.zeros(self.tree_model.tree_.n_features)
        tree_ = self.tree_model.tree_
        for node in node_list:
            if self.tree_model.tree_.children_left[node] != -1:
                node_left = self.tree_model.tree_.children_left[node]
                node_right = self.tree_model.tree_.children_right[node]
                gini[tree_.feature[node]] += tree_.weighted_n_node_samples[node] * tree_.impurity[node] \
                                             - tree_.weighted_n_node_samples[node_left] * tree_.impurity[node_left] \
                                             - tree_.weighted_n_node_samples[node_right] * tree_.impurity[node_right]
        normalizer = np.sum(gini)
        if normalizer > 0.0:
            gini /= normalizer

        return gini

    def get_max_depth(self):
        """
        Gets the maximum depth of the tree.

        :return: The maximum depth of the tree.
        :rtype: int
        """
        return self.tree_model.maxDepth

    def get_score(self):
        """
        Scores the model. For a classification tree, gets the mean accuracy. For a regression tree, gets 
        the coefficient of determination (R\ :sup:`2`).

        :return: The score of the model
        :rtype: float
        """
        X = pd.DataFrame(self.x_data, columns=self.feature_names)
        return self.tree_model.score(X, self.y_data)

    def get_min_samples_leaf(self):
        """
        Gets the minimum number of samples required to be in a leaf node.

        :return: The minimum number of samples required to be in a leaf node.
        :rtype: int
        """
        return self.tree_model.nodesizeSpl

    def shouldGoLeftAtSplit(self, id, x):
        """
        Finds out whether an observation should go to the left child node from a given node.

        :param id: The id of the node.
        :type id: *int*
        :param x: A given observation.
        :type x: *int, float*
        :return: *True* if the given observation should go to the left child node of the current node.
        :rtype: bool
        """
        return x < self.get_node_split(id)