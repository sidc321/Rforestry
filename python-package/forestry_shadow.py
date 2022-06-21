from collections import defaultdict
from typing import List, Mapping

import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight

from dtreeviz.models.shadow_decision_tree import ShadowDecTree


class ShadowForestryTree(ShadowDecTree):
    def __init__(self, tree_model,
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 tree_id = 0,
                 class_names: (List[str], Mapping[int, str]) = None):

        self.node_to_samples = None
        self.tree_id = tree_id
        tree_model.translate_tree_python(tree_id)
        super().__init__(tree_model, x_data, y_data, feature_names, target_name, class_names)

    def is_fit(self):
        return getattr(self.tree_model, 'forest') is not None

    def is_classifier(self):
        ## TODO: Change if we implement classification as a different method
        return False

    def get_class_weights(self):
        if self.is_classifier():
            unique_target_values = np.unique(self.y_data)
            return compute_class_weight(self.tree_model.class_weight, classes=unique_target_values, y=self.y_data)

    def get_thresholds(self):
        return self.tree_model.Py_forest[self.tree_id]["threshold"]

    def get_features(self):
        return self.tree_model.Py_forest[self.tree_id]["feature"]

    def criterion(self):
        return 'SQUARED_ERROR'

    def get_class_weight(self):
        if self.is_classifier():
            return self.tree_model.class_weight

    def nclasses(self):
        return 1

    def classes(self):
        if self.is_classifier():
            return self.tree_model.classes_

    #TODO: Implement this
    def get_node_samples(self):
        if self.node_to_samples is not None:
            return self.node_to_samples

        dec_paths = self.tree_model.decision_path(self.x_data, self.tree_id)

        # each sample has path taken down tree
        node_to_samples = defaultdict(list)
        for sample_i, dec in enumerate(dec_paths):
            for node_id in dec:
                node_to_samples[node_id].append(sample_i)

        self.node_to_samples = node_to_samples
        return node_to_samples

    #TODO: Implement this
    def get_split_samples(self, id):
        samples = np.array(self.get_node_samples()[id])
        node_X_data = self.x_data[samples, self.get_node_feature(id)]
        split = self.get_node_split(id)

        left = np.nonzero(node_X_data <= split)[0]
        right = np.nonzero(node_X_data > split)[0]

        return left, right

    def get_root_edge_labels(self):
        return ["&le;", "&gt;"]

    def get_node_nsamples(self, id):
        return len(self.get_node_samples()[id])

    def get_children_left(self):
        return self.tree_model.Py_forest[self.tree_id]["children_left"]

    def get_children_right(self):
        return self.tree_model.Py_forest[self.tree_id]["children_right"]

    def get_node_split(self, id) -> (int, float):
        return self.tree_model.Py_forest[self.tree_id]["threshold"][id]

    def get_node_feature(self, id) -> int:
        return self.tree_model.Py_forest[self.tree_id]["feature"][id]

    def get_node_nsamples_by_class(self, id):
        if self.is_classifier():
            return self.tree_model.Py_forest[self.tree_id]["value"][id][0]

    def get_prediction(self, id):
        if self.is_classifier():
            counts = self.tree_model.Py_forest[self.tree_id]["value"][id][0]
            return np.argmax(counts)
        else:
            return self.tree_model.Py_forest[self.tree_id]["value"][id]

    def nnodes(self):
        return len(self.tree_model.Py_forest[self.tree_id]["feature"])

    def get_node_criterion(self, id):
        return self.tree_model.tree_.impurity[id]

    def get_feature_path_importance(self, node_list):
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
        return self.tree_model.maxDepth

    def get_score(self):
        X = pd.DataFrame(self.x_data, columns=self.feature_names)
        return self.tree_model.score(X, self.y_data)

    def get_min_samples_leaf(self):
        return self.tree_model.nodesizeSpl

    def shouldGoLeftAtSplit(self, id, x):
        return x < self.get_node_split(id)