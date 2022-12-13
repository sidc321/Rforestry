import dataclasses
import math
import os
import pickle  # nosec B403 - 'Consider possible security implications associated with pickle'
import sys
import warnings
from pathlib import Path
from random import randrange
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import ConfigDict, StrictBool, StrictFloat, StrictInt, confloat, conint, validate_arguments
from pydantic.dataclasses import dataclass
from sklearn.model_selection import LeaveOneOut

from . import extension, preprocessing  # type: ignore
from .processed_dta import ProcessedDta
from .fit_validator import FitValidator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, smart_union=True, validate_all=True))
class RandomForest:
    """
    The Random Forest Regressor class.

    :param ntree: The number of trees to grow in the forest.
    :type ntree: *int, optional, default=500*
    :param replace: An indicator of whether sampling of the training data is done with replacement.
    :type replace: *bool, optional, default=True*
    :param sampsize: The size of total samples to draw for the training data. If sampling with replacement, the default
     value is the length of the training data. If sampling without replacement, the default value is two-thirds of the
     length of the training data.
    :type sampsize: *int, optional*
    :param sample_fraction: If this is given, then sampsize is ignored and set to
     be ``round(len(y) * sample_fraction)``. It must be a real number between 0 and 1.
    :type sample_fraction: *float, optional*
    :param mtry: The number of variables randomly selected at each split point. The default value is set to be
     one-third of the total number of features of the training data.
    :type mtry: *int, optional*
    :param nodesizeSpl: Minimum observations contained in terminal nodes.
    :type nodesizeSpl: *int, optional, default=5*
    :param nodesizeAvg: Minimum size of terminal nodes for averaging dataset.
    :type nodesizeAvg: *int, optional, default=5*
    :param nodesizeStrictSpl: Minimum observations to follow strictly in terminal nodes.
    :type nodesizeStrictSpl: *int, optional, default=1*
    :param nodesizeStrictAvg: The minimum size of terminal nodes for averaging data set to follow when predicting.
     No splits are allowed that result in nodes with observations less than this parameter.
     This parameter enforces overlap of the averaging data set with the splitting set when training.
     When using honesty, splits that leave less than nodesizeStrictAvg averaging
     observations in either child node will be rejected, ensuring every leaf node
     also has at least nodesizeStrictAvg averaging observations.
    :type nodesizeStrictAvg: *int, optional, default=1*
    :param minSplitGain: Minimum loss reduction to split a node further in a tree.
    :type minSplitGain: *float, optional, default=0*
    :param maxDepth: Maximum depth of a tree.
    :type maxDepth: *int, optional, default=99*
    :param interactionDepth: All splits at or above interaction depth must be on variables
     that are not weighting variables (as provided by the interactionVariables argument in fit).
    :type interactionDepth: *int, optional, default=maxDepth*
    :param splitratio: Proportion of the training data used as the splitting dataset.
     It is a ratio between 0 and 1. If the ratio is 1 (the default), then the splitting
     set uses the entire data, as does the averaging set---i.e., the standard Breiman RF setup.
     If the ratio is 0, then the splitting data set is empty, and the entire dataset is used
     for the averaging set (This is not a good usage, however, since there will be no data available for splitting).
    :type splitratio: *double, optional, default=1*
    :param OOBhonest: In this version of honesty, the out-of-bag observations for each tree
     are used as the honest (averaging) set. This setting also changes how predictions
     are constructed. When predicting for observations that are out-of-sample
     ``(predict(..., aggregation = "average"))``, all the trees in the forest
     are used to construct predictions. When predicting for an observation that was in-sample
     ``(predict(..., aggregation = "oob"))``, only the trees for which that observation
     was not in the averaging set are used to construct the prediction for that observation.
     *aggregation="oob"* (out-of-bag) ensures that the outcome value for an observation
     is never used to construct predictions for a given observation even when it is in sample.
     This property does not hold in standard honesty, which relies on an asymptotic
     subsampling argument. By default, when *OOBhonest=True*, the out-of-bag observations
     for each tree are resamples with replacement to be used for the honest (averaging)
     set. This results in a third set of observations that are left out of both
     the splitting and averaging set, we call these the double out-of-bag (doubleOOB)
     observations. In order to get the predictions of only the trees in which each
     observation fell into this doubleOOB set, one can run ``predict(... , aggregation = "doubleOOB")``.
     In order to not do this second bootstrap sample, the doubleBootstrap flag can
     be set to *False*.
    :type OOBhonest: *bool, optional, default=False*
    :param doubleBootstrap: The doubleBootstrap flag provides the option to resample
     with replacement from the out-of-bag observations set for each tree to construct
     the averaging set when using OOBhonest. If this is *False*, the out-of-bag observations
     are used as the averaging set. By default this option is *True* when running *OOBhonest=True*.
     This option increases diversity across trees.
    :type doubleBootstrap: *bool, optional, default=OOBhonest*
    :param seed: Random number generator seed. The default value is a random integer.
    :type seed: *int, optional*
    :param verbose: Indicator to train the forest in verbose mode.
    :type verbose: *bool, optional, default=False*
    :param nthread: Number of threads to train and predict the forest. The default
     number is 0 which represents using all cores.
    :type nthread: *int, optional, default=0*
    :param splitrule: Only variance is implemented at this point and, it
     specifies the loss function according to which the splits of random forest
     should be made.
    :type splitrule: *str, optional, default='variance'*
    :param middleSplit: Indicator of whether the split value is takes the average of two feature
     values. If *False*, it will take a point based on a uniform distribution
     between two feature values.
    :type middleSplit: *bool, optional, default=False*
    :param maxObs: The max number of observations to split on. The default is the number of observations.
    :type maxObs: *int, optional*
    :param linear: Indicator that enables Ridge penalized splits and linear aggregation
     functions in the leaf nodes. This is recommended for data with linear outcomes.
     For implementation details, see: https://arxiv.org/abs/1906.06463.
    :type linear: *bool, optional, default=False*
    :param minTreesPerGroup: The number of trees which we make sure have been created leaving
     out each group. This is 0 by default, so we will not give any special treatment to
     the groups when sampling, however if this is set to a positive integer, we
     modify the bootstrap sampling scheme to ensure that exactly that many trees
     have the group left out. We do this by, for each group, creating *minTreesPerGroup*
     trees which are built on observations sampled from the set of training observations
     which are not in the current group. This means we create at least ``len(groups)*minTreesPerGroup``
     trees for the forest. If ``ntree>len(groups)*minTreesPerGroup``, we create
     ``max(len(groups)*minTreesPerGroup,ntree)`` total trees, in which at least *minTreesPerGroup*
     are created leaving out each group. For debugging purposes, these group sampling
     trees are stored at the end of the Python forest, in blocks based on the left out group.
    :type minTreesPerGroup: *int, optional, default=0*
    :param monotoneAvg: This is a flag that indicates whether or not monotonic
     constraints should be enforced on the averaging set in addition to the splitting set.
     This flag is meaningless unless both honesty and monotonic constraints are in use.
    :type monotoneAvg: *bool, optional, default=False*
    :param overfitPenalty: Value to determine how much to penalize the magnitude
     of coefficients in ridge regression when using linear splits.
    :type overfitPenalty: *float, optional, default=1*
    :param scale: A parameter which indicates whether or not we want to scale and center
     the covariates and outcome before doing the regression. This can help with
     stability, so the default is *True*.
    :type scale: *bool, optional, default=True*
    :param doubleTree: Indicator of whether the number of trees is doubled as averaging and splitting
     data can be exchanged to create decorrelated trees.
    :type doubleTree: *bool, optional, default=False*

    :ivar processed_dta: A dictionary containing information about the data after it has been preprocessed.
     *processed_dta* has the following entries:

     * processed_x (*pandas.DataFrame*) - The processed feature matrix.

     * y (*numpy.array of shape[nrows,]*) - The processed target values.

     * categoricalFeatureCols (*numpy.array*) - An array of the indices of the categorical features in the feature matrix.

      .. note::
        In order for the program to recognize a feature as categorical, it **must** be converted into a
        `Pandas categorical data type <https://pandas.pydata.org/docs/user_guide/categorical.html#>`_. The
        simplest way to do it is to use::

            df['categorical'] = df['categorical'].astype('category')

        Check out the :ref:`Handling Categorical Data <categorical>` section for an example of how to use categorical features.

     * categoricalFeatureMapping (*list[dict]*) - For each categorical feature, the data is encoded into numeric represetation. Those encodings are saved in *categoricalFeatureMapping*. *categoricalFeatureMapping[i]* has the following entries:

        * categoricalFeatureCol (*int*) - The index of the current categorical feature column.

        * uniqueFeatureValues (*list*) - The categories of the current categorical feature.

        * numericFeatureValues (*numpy.array*) - The categories of the current categorical feature encoded into numeric represetation.

     * featureWeights (*numpy.array of shape[ncols]*) - an array of sampling probabilities/weights for each feature used when subsampling *mtry* features at each node. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * featureWeightsVariables (*numpy.array*) - Indices of the features which weight more than ``max(featureWeights)*0.001``.

     * deepFeatureWeights (*numpy.array of shape[ncols]*) - Used in place of *featureWeights* for splits below *interactionDepth*. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * deepFeatureWeightsVariables (*numpy.array*) - Indices of the features which weight more than ``max(deepFeatureWeights)*0.001``.

     * observationWeights (*numpy.array of shape[nrows]*) - Denotes the weights for each training observation that determine how likely the observation is to be selected in each bootstrap sample. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * symmetric (*numpy.array  of shape[ncols]*) - Used for the experimental feature which imposes strict symmetric marginal structure on the predictions of the forest through only selecting symmetric splits with symmetric aggregation functions. It's a numpy array of size *ncols* consisting of 0-s and 1-s, with 1 denoting the features to enforce symmetry on. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * monotonicConstraints (*numpy.array of shape[ncols]*) - An array of size *ncol* specifying monotonic relationships between the continuous features and the outcome. Its entries are in -1, 0, 1, in which 1 indicates an increasing monotonic relationship, -1 indicates a decreasing monotonic relationship, and 0 indicates no constraint. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * linearFeatureCols (*numpy.array*) - An array containing the indices of which features to split linearly on when using linear penalized splits. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * groupsMapping (*dict*) - Contains informtion about the groups of the training observations. Has the following entries:

        * groupValue (*pandas.Index*) - The categories of the groups.

        * groupNumericValue (*numpy.array*) - The categories of the groups encoded into numeric represetation

     * groups (*pandas.Series(..., dtype='category')*) - Specifies the group membership of each training observation. Check out :meth:`fit() <forestry.RandomForest.fit>` fot more details.

     * colMeans (*numpy.array of shape[ncols]*) - The mean value of each column.

     * colSd (*numpy.array of shape[ncols]*) - The standard deviation of each column.

     * hasNas (*bool*) - Specifies whether the feature matrix contains missing observations or not.

     * nObservations (*int*) - The number of observations in the training data.

     * numColumns (*int*) - The number of features in the training data.

     * featNames (*numpy.array of shape[ncols]*) - The names of the features used for training.

     Note that **all** of the entries in processed_dta are set to ``None`` during initialization. They are only assigned a value after :meth:`fit() <forestry.RandomForest.fit>` is called.

    :vartype processed_dta: dict


    .. _translate-label:

    :ivar Py_forest: For any tree *i* in the forest, *Py_forest[i]* is a dictionary which gives access to the underlying structrure of that tree. *Py_forest[i]* has the following entries:

     * children_right (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *children_right[id]* gives the id of the right child of that node. If leaf node, *children_right[id]* is *-1*.

     * children_left (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *children_left[id]* gives the id of the left child of that node. If leaf node, *children_left[id]* is *-1*.

     * feature (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *feature[id]* gives the index of the splitting feature in that node. If leaf node, *feature[id]* is the negative number of observations in the averaging set of that node.

     * n_node_samples (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *feature[id]* gives the number of observations in the averaging set of that node.

     * threshold (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, *threshold[id]* gives the splitting point (threshold) of the split in that node. If leaf node, *threshold[id]* is *0.0*.

     * values (*numpy.array of shape[number of nodes in the tree,]*) - For a node with a given *id*, if that node is a leaf node, *values[id]* gives the prediction made by that node. Otherwise, *values[id]* is *0.0*.

     .. note::
        When a *RandomForest* is initialized, *Py_forest* is set to a list of *ntree* empty dictionaries.
        In order to populate those dictionaries, one must use the :meth:`translate_tree_python() <forestry.RandomForest.translate_tree_python>` method.

    :vartype Py_forest: list[dict]
    :ivar forest: A ctypes pointer to the *forestry* object in C++. It is initially set to *None* and updated only
     after :meth:`fit() <forestry.RandomForest.fit>` is called.
    :vartype forest: ctypes.c_void_p
    :ivar dataframe: A ctypes pointer to the *DataFrame* object in C++. It is initially set to *None* and updated only
     after :meth:`fit() <forestry.RandomForest.fit>` is called.
    :vartype dataframe: ctypes.c_void_p

    """

    ntree: conint(gt=0, strict=True) = 500
    replace: StrictBool = True
    sampsize: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    sample_fraction: Optional[Union[conint(gt=0, strict=True), confloat(gt=0, strict=True)]] = None
    mtry: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    nodesize_spl: conint(gt=0, strict=True) = 5
    nodesize_avg: conint(gt=0, strict=True) = 5
    nodesize_strict_spl: conint(gt=0, strict=True) = 1
    nodesize_strict_avg: conint(gt=0, strict=True) = 1
    min_split_gain: confloat(ge=0) = 0
    max_depth: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    interaction_depth: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    splitratio: confloat(ge=0, le=1) = 1.0
    oob_honest: StrictBool = False
    double_bootstrap: Optional[StrictBool] = None  # Add a default value.
    seed: conint(ge=0, strict=True) = randrange(1001)  # nosec B311
    verbose: StrictBool = False
    nthread: conint(ge=0, strict=True) = 0
    splitrule: str = "variance"
    middle_split: StrictBool = False
    max_obs: Optional[conint(gt=0, strict=True)] = None  # Add a default value.
    linear: StrictBool = False
    min_trees_per_group: conint(ge=0, strict=True) = 0
    monotone_avg: StrictBool = False
    overfit_penalty: Union[StrictInt, StrictFloat] = 1
    scale: StrictBool = False
    double_tree: StrictBool = False

    forest: Optional[pd.DataFrame] = dataclasses.field(default=None, init=False)
    dataframe: Optional[pd.DataFrame] = dataclasses.field(default=None, init=False)
    processed_dta: Optional[ProcessedDta] = dataclasses.field(default=None, init=False)
    py_forest: List[Dict] = dataclasses.field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.nthread > os.cpu_count():
            raise ValueError("nthread cannot exceed total cores in the computer: " + str(os.cpu_count()))

        if self.min_split_gain > 0 and not self.linear:
            raise ValueError("min_split_gain cannot be set without setting linear to be true.")

        if self.double_bootstrap is None:
            self.double_bootstrap = self.oob_honest

        if self.oob_honest and (self.splitratio != 1):
            warnings.warn("oob_honest is set to true, so we will run OOBhonesty rather than standard honesty.")
            self.splitratio = 1

        if self.oob_honest and not self.replace:
            warnings.warn("replace must be set to TRUE to use OOBhonesty, setting this to True now")
            self.replace = True

        if self.double_tree and self.splitratio in (0, 1):
            warnings.warn("Trees cannot be doubled if splitratio is 1. We have set double_tree to False.")
            self.double_tree = False

        if (
            self.interaction_depth is not None
            and self.max_depth is not None
            and self.interaction_depth > self.max_depth
        ):
            warnings.warn(
                "interaction_depth cannot be greater than max_depth. We have set interaction_depth to max_depth."
            )
            self.interaction_depth = self.max_depth

    @validate_arguments
    def _get_seed(self, seed: Optional[int]) -> int:
        if seed is None:
            return self.seed
        if (not isinstance(seed, int)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        return seed

    def _set_nodesize_strict(self) -> None:

        # if the splitratio is 1, then we use adaptive rf and avgSampleSize is
        # equal to the total sampsize

        if self.splitratio in (0, 1):
            split_sample_size = self.sampsize
            avg_sample_size = self.sampsize
        else:
            split_sample_size = self.splitratio * self.sampsize
            avg_sample_size = math.floor(self.sampsize - split_sample_size)
            split_sample_size = math.floor(split_sample_size)

        if self.nodesize_strict_spl > split_sample_size:
            warnings.warn(
                "nodesizeStrictSpl cannot exceed splitting sample size. ",
                "We have set nodesizeStrictSpl to be the maximum.",
            )
            self.nodesize_strict_spl = split_sample_size

        if self.nodesize_strict_avg > avg_sample_size:
            warnings.warn(
                "nodesizeStrictAvg cannot exceed averaging sample size. ",
                "We have set nodesizeStrictAvg to be the maximum.",
            )
            self.nodesize_strict_avg = avg_sample_size

        if self.double_tree:
            if self.nodesize_strict_avg > split_sample_size:
                warnings.warn(
                    "nodesizeStrictAvg cannot exceed splitting sample size. ",
                    "We have set nodesizeStrictAvg to be the maximum.",
                )
                self.nodesize_strict_avg = split_sample_size
            if self.nodesize_strict_spl > avg_sample_size:
                warnings.warn(
                    "nodesize_strict_spl cannot exceed averaging sample size. ",
                    "We have set nodesize_strict_spl to be the maximum.",
                )
                self.nodesize_strict_spl = avg_sample_size

    def _get_weights_variables(self, weights: np.ndarray) -> np.ndarray:
        weights_variables = [i for i in range(weights.size) if weights[i] > max(weights) * 0.001]
        if len(weights_variables) < self.mtry:
            raise ValueError("mtry is too large. Given the feature weights, can't select that many features.")

        weights_variables = np.array(weights_variables, dtype=np.ulonglong)
        return weights_variables

    def _get_groups_mapping_and_vector(self, x: pd.DataFrame, groups: Optional[pd.Series]) -> Tuple[dict, pd.Series]:
        nrow, _ = x.shape
        groups_mapping = {}
        if groups is not None:
            groups_mapping["groupValue"] = groups.cat.categories
            groups_mapping["groupNumericValue"] = np.arange(len(groups.cat.categories))

            group_vector = pd.to_numeric(groups)

            # Print warning if the group number and minTreesPerGroup results in a large forest
            if self.min_trees_per_group > 0 and len(groups.cat.categories) * self.min_trees_per_group > 2000:
                warnings.warn(
                    "Using "
                    + str(len(groups.cat.categories))
                    + " groups with "
                    + str(self.min_trees_per_group)
                    + " trees per group will train "
                    + str(len(groups.cat.categories) * self.min_trees_per_group)
                    + " trees in the forest"
                )

        else:
            group_vector = np.zeros(nrow, dtype=np.ulonglong)
        return groups_mapping, group_vector

    @FitValidator
    def fit(
        self,
        x: Union[pd.DataFrame, pd.Series, List],
        y: np.ndarray,
        *,
        interaction_variables: Optional[List] = None,
        feature_weights: Optional[np.ndarray] = None,
        deep_feature_weights: Optional[np.ndarray] = None,
        observation_weights: Optional[np.ndarray] = None,
        symmetric: Optional[np.ndarray] = None,  # Add a default value.
        lin_feats: Optional[Union[np.ndarray, List]] = None,  # Add a default value.
        monotonic_constraints: Optional[np.ndarray] = None,  # Add a default value.
        groups: Optional[pd.Series] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Trains all the trees in the forest.

        :param x: The feature matrix.
        :type x: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nrows, ncols]*
        :param y: The target values.
        :type y: *array_like of shape [nrows,]*
        :param interactionVariables: Indices of weighting variables.
        :type interactionVariables: *array_like, optional, default=[]*
        :param featureWeights: a list of sampling probabilities/weights for each
         feature used when subsampling *mtry* features at each node above or at *interactionDepth*.
         The default is to use uniform probabilities.
        :type featureWeights: *array_like of shape [ncols,], optional*
        :param deepFeatureWeights: Used in place of *featureWeights* for splits below *interactionDepth*.
         The default is to use uniform probabilities.
        :type deepFeatureWeights: *array_like of shape [ncols,], optional*
        :param observationWeights: Denotes the weights for each training observation
         that determine how likely the observation is to be selected in each bootstrap sample.
         The default is to use uniform probabilities. This option is not allowed when sampling is
         done without replacement.
        :type observationWeights: *array_like of shape [nrows,], optional*
        :param symmetric: Used for the experimental feature which imposes strict symmetric
         marginal structure on the predictions of the forest through only selecting
         symmetric splits with symmetric aggregation functions. Should be a list of size *ncols* with a single
         1 entry denoting the feature to enforce symmetry on. Defaults to all zeroes.
         For version >= 0.9.0.83, we experimentally allow more than one feature to
         enforce symmetry at a time. This should only be used for a small number of
         features as it has a runtime that is exponential in the number of symmetric
         features - ``O(N*2^|S|)`` - where S is the set of symmetric features).
        :type symmetric: *array_like of shape [ncols,], optional*
        :param linFeats: A list containing the indices of which features to split
         linearly on when using linear penalized splits (defaults to use all numerical features).
        :type linFeats: *array_like, optional*
        :param monotonicConstraints: Specifies monotonic relationships between the continuous
         features and the outcome. Supplied as a list of length *ncol* with entries in
         1, 0, -1, with 1 indicating an increasing monotonic relationship, -1 indicating
         a decreasing monotonic relationship, and 0 indicating no constraint.
         Constraints supplied for categorical variable will be ignored. Defaults to all 0-s (no constraints).
        :type monotonicConstraints: *array_like of shape [ncols,], optional*
        :param groups: A pandas categorical Seires specifying the group membership of each training observation.
         These groups are used in the aggregation when doing out of bag predictions in
         order to predict with only trees where the entire group was not used for aggregation.
         This allows the user to specify custom subgroups which will be used to create
         predictions which do not use any data from a common group to make predictions for
         any observation in the group. This can be used to create general custom
         resampling schemes, and provide predictions consistent with the Out-of-Group set.
        :type groups: *pandas.Categorical(...), pandas.Series(..., dtype="category"),
         or other pandas categorical dtypes, optional, default=None*
        :param seed: Random number generator seed. The default value is the *RandomForest* seed.
        :type seed: *int, optional*
        :rtype: None
        """

        # Make sure that all the parameters exist when passed to RandomForest

        feat_names = preprocessing.get_feat_names(x)

        x = (pd.DataFrame(x)).copy()
        y = (np.array(y, dtype=np.double)).copy()

        nrow, ncol = x.shape

        if self.max_depth is None:
            self.max_depth = round(nrow / 2) + 1

        if self.interaction_depth is None:
            self.interaction_depth = self.max_depth

        if self.max_obs is None:
            self.max_obs = y.size

        self.sampsize = preprocessing.get_sampsize(self, x)
        self.mtry = preprocessing.get_mtry(self, x)

        if any(i != 0 for i in symmetric) and self.scale:
            warnings.warn(
                "As symmetry is implementing pseudo outcomes, this causes problems when the Y values are scaled. "
                "Setting scale = False"
            )
            # for now don't scale when we run symmetric splitting since we use pseudo outcomes
            # and want to retain the scaling of Y
            self.scale = False

        self._set_nodesize_strict()

        feature_weights_variables = self._get_weights_variables(feature_weights)
        deep_feature_weights_variables = self._get_weights_variables(deep_feature_weights)

        feature_weights /= np.sum(feature_weights)
        deep_feature_weights /= np.sum(deep_feature_weights)
        if self.replace:
            observation_weights /= np.sum(observation_weights)

        groups_mapping, group_vector = self._get_groups_mapping_and_vector(x, groups)

        (
            processed_x,
            categorical_feature_cols,
            categorical_feature_mapping,
        ) = preprocessing.preprocess_training(x, y)

        if categorical_feature_cols.size != 0:
            monotonic_constraints[categorical_feature_cols] = 0

        col_means = col_sd = np.repeat(0.0, ncol + 1)
        if self.scale:
            processed_x, y, col_means, col_sd = preprocessing.scale(x, y, processed_x, categorical_feature_cols)

        # Get the symmetric feature if one is set
        symmetric_index = -1
        idxs = np.where(symmetric > 0)[0]
        if idxs.size != 0:
            symmetric_index = idxs[0]

        # cpp linking
        processed_x.reset_index(drop=True, inplace=True)

        self.dataframe: pd.DataFrame = extension.get_data(
            np.ascontiguousarray(pd.concat([processed_x, pd.Series(y)], axis=1).values[:, :], np.double).ravel(),
            categorical_feature_cols,
            categorical_feature_cols.size,
            lin_feats,
            lin_feats.size,
            feature_weights,
            feature_weights_variables,
            feature_weights_variables.size,
            deep_feature_weights,
            deep_feature_weights_variables,
            deep_feature_weights_variables.size,
            observation_weights,
            monotonic_constraints,
            group_vector,
            self.monotone_avg,
            symmetric,
            symmetric.size,
            nrow,
            ncol + 1,
            self._get_seed(seed),
        )

        self.forest: pd.DataFrame = extension.train_forest(
            self.dataframe,
            self.ntree,
            self.replace,
            self.sampsize,
            self.splitratio,
            self.oob_honest,
            self.double_bootstrap,
            self.mtry,
            self.nodesize_spl,
            self.nodesize_avg,
            self.nodesize_strict_spl,
            self.nodesize_strict_avg,
            self.min_split_gain,
            self.max_depth,
            self.interaction_depth,
            self._get_seed(seed),
            self.nthread,
            self.verbose,
            self.middle_split,
            self.max_obs,
            self.min_trees_per_group,
            x.isnull().values.any(),
            self.linear,
            symmetric_index != -1,
            self.overfit_penalty,
            self.double_tree,
        )

        # Update the fields
        self.processed_dta = ProcessedDta(
            processed_x=processed_x,
            y=y,
            categorical_feature_cols=categorical_feature_cols,
            categorical_feature_mapping=categorical_feature_mapping,
            feature_weights=feature_weights,
            feature_weights_variables=feature_weights_variables,
            deep_feature_weights=deep_feature_weights,
            deep_feature_weights_variables=deep_feature_weights_variables,
            observation_weights=observation_weights,
            symmetric=symmetric,
            monotonic_constraints=monotonic_constraints,
            linear_feature_cols=lin_feats,
            groups_mapping=groups_mapping,
            groups=groups,
            col_means=col_means,
            col_sd=col_sd,
            has_nas=x.isnull().values.any(),
            n_observations=nrow,
            num_columns=ncol,
            feat_names=feat_names,
        )

    def _get_nthread(self, nthread: Optional[int]) -> int:
        if nthread is None:
            return self.nthread
        return nthread

    def _get_test_data(self, newdata: Optional[pd.DataFrame]) -> np.ndarray:
        if newdata is None:
            return np.ascontiguousarray(self.processed_dta.processed_x.values[:, :], np.double).ravel()

        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )
        return np.ascontiguousarray(processed_x.values[:, :], np.double).ravel()

    def _get_n_preds(self, newdata: Optional[pd.DataFrame]) -> int:
        if newdata is None:
            return self.processed_dta.n_observations

        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )
        return len(processed_x.index)

    def _aggregation_oob(
        self, newdata: Optional[pd.DataFrame], exact: Optional[bool], return_weight_matrix: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        if newdata is not None and self.processed_dta.n_observations != len(newdata.index):
            warnings.warn("Attempting to do OOB predictions on a dataset which doesn't match the training data!")
            return None

        n_preds = self._get_n_preds(newdata)
        n_weight_matrix = n_preds * self.processed_dta.n_observations if return_weight_matrix else 0

        return extension.predict_oob_forest(
            self.forest,
            self.dataframe,
            self._get_test_data(newdata),
            False,
            preprocessing.predict_exact(newdata, exact),
            return_weight_matrix,
            self.verbose,
            n_preds,
            n_weight_matrix,
        )

    def _aggregation_double_oob(
        self, newdata: Optional[pd.DataFrame], exact: Optional[bool], return_weight_matrix: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.double_bootstrap:
            raise ValueError(
                "Attempting to do double OOB predictions with a forest that was not trained with doubleBootstrap = True"
            )

        if newdata is None:
            double_oob = True
        else:
            double_oob = False
            processed_x = preprocessing.preprocess_testing(
                newdata,
                self.processed_dta.categorical_feature_cols,
                self.processed_dta.categorical_feature_mapping,
            )
            if len(processed_x.index) != self.processed_dta.n_observations:
                raise ValueError("Attempting to do OOB predictions on a dataset which doesn't match the training data!")

        n_preds = self._get_n_preds(newdata)
        n_weight_matrix = n_preds * self.processed_dta.n_observations if return_weight_matrix else 0

        return extension.predict_oob_forest(
            self.forest,
            self.dataframe,
            self._get_test_data(newdata),
            double_oob,
            preprocessing.predict_exact(newdata, exact),
            return_weight_matrix,
            self.verbose,
            n_preds,
            n_weight_matrix,
        )

    def _aggregation_coefs(
        self, newdata: pd.DataFrame, exact: Optional[bool], seed: int, nthread: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.linear:
            raise ValueError("Aggregation can only be linear with setting the parameter linear = True.")
        if newdata is None:
            raise ValueError("When using an aggregation that is not oob or doubleOOB, one must supply newdata")
        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )

        return extension.predict_forest(
            self.forest,
            self.dataframe,
            np.ascontiguousarray(processed_x.values[:, :], np.double).ravel(),
            seed,
            nthread,
            preprocessing.predict_exact(newdata, exact),
            False,
            True,
            False,
            np.zeros(self.ntree, dtype=np.ulonglong),
            len(processed_x.index),
            self._get_n_preds(newdata),
            0,
            self.processed_dta.n_observations * (self.processed_dta.linear_feature_cols.size + 1),
        )

    def _aggregation_fallback(
        self,
        newdata: pd.DataFrame,
        aggregation: str,
        exact: Optional[bool],
        seed: int,
        nthread: int,
        return_weight_matrix: bool,
        trees: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if newdata is None:
            raise ValueError("When using an aggregation that is not oob or doubleOOB, one must supply newdata")
        processed_x = preprocessing.preprocess_testing(
            newdata,
            self.processed_dta.categorical_feature_cols,
            self.processed_dta.categorical_feature_mapping,
        )
        exact = preprocessing.predict_exact(newdata, exact)
        tree_weights = np.zeros(self.ntree, dtype=np.ulonglong)
        # We can only use tree aggregations if exact = True and aggregation = "average"
        if trees is not None:
            if not exact or aggregation != "average":
                raise ValueError("When using tree indices, we must have exact = True and aggregation = 'average' ")

            if any((not isinstance(i, (int, np.integer))) or (i < -self.ntree) or (i >= self.ntree) for i in trees):
                raise ValueError("trees must contain indices which are integers between -ntree and ntree-1")

            # If trees are being used, we need to convert them into a weight vector
            for tree in trees:
                tree_weights[tree] += 1
            use_weights = True
        else:
            use_weights = False

        n_preds = self._get_n_preds(newdata)
        n_weight_matrix = n_preds * self.processed_dta.n_observations if return_weight_matrix else 0
        print(seed, nthread)

        return extension.predict_forest(
            self.forest,
            self.dataframe,
            np.ascontiguousarray(processed_x.values[:, :], np.double).ravel(),
            seed,
            nthread,
            exact,
            return_weight_matrix,
            False,
            use_weights,
            tree_weights,
            len(processed_x.index),
            n_preds,
            n_weight_matrix,
            0,
        )

    def predict(
        self,
        newdata: Optional[Union[pd.DataFrame, pd.Series, List]] = None,
        aggregation: str = "average",
        seed: Optional[int] = None,
        nthread: Optional[int] = None,
        exact: Optional[bool] = None,
        trees: Optional[np.ndarray] = None,
        return_weight_matrix: bool = False,
    ) -> Union[np.ndarray, dict]:
        """
        Return the prediction from the forest.

        :param newdata: Testing predictors.
        :type newdata: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols], deffault=None*
        :param aggregation: How the individual tree predictions are aggregated:
         'average' returns the mean of all trees in the forest; 'terminalNodes' also returns
         the weightMatrix, as well as "terminalNodes" - a matrix where
         the i-th entry of the j-th column is the index of the leaf node to which the
         i-th observation is assigned in the j-th tree; and "sparse" - a matrix
         where the ioth entry in the j-th column is 1 if the ith observation in
         newdata is assigned to the j-th leaf and 0 otherwise. In each tree the
         leaves are indexed using a depth first ordering, and, in the "sparse"
         representation, the first leaf in the second tree has column index one more than
         the number of leaves in the first tree and so on. So, for example, if the
         first tree has 5 leaves, the sixth column of the "sparse" matrix corresponds
         to the first leaf in the second tree.
         'oob' returns the out-of-bag predictions for the forest. We assume
         that the ordering of the observations in newdata have not changed from
         training. If the ordering has changed, we will get the wrong OOB indices.
         'doubleOOB' is an experimental flag, which can only be used when *OOBhonest=True*
         and *doubleBootstrap=True*. When both of these settings are on, the
         splitting set is selected as a bootstrap sample of observations and the
         averaging set is selected as a bootstrap sample of the observations which
         were left out of bag during the splitting set selection. This leaves a third
         set which is the observations which were not selected in either bootstrap sample.
         For each observation, this predict flag gives the predictions using only the trees
         in which the observation fell into this third set (so was neither a splitting
         nor averaging example).
         'coefs' is an aggregation option which works only when linear aggregation
         functions have been used. This returns the linear coefficients for each
         linear feature which were used in the leaf node regression of each predicted point.
        :type aggregation: *str, optional, default='average'*
        :param seed: Random number generator seed. The default value is the *RandomForest* seed.
        :type seed: *int, optional*
        :param nthread: The number of threads with which to run the predictions with.
         This will default to the number of threads with which the forest was trained
         with.
        :type nthread: *int, optional*
        :param exact: This specifies whether the forest predictions should be aggregated
         in a reproducible ordering. Due to the non-associativity of floating point
         addition, when we predict in parallel, predictions will be aggregated in
         varied orders as different threads finish at different times.
         By default, exact is *True* unless ``N>100,000`` or a custom aggregation
         function is used.
        :type exact: *bool, optional*
        :param trees: A list of indices in the range *[0, ntree)*, which tells
         predict which trees in the forest to use for the prediction. Predict will by
         default take the average of all trees in the forest, although this flag
         can be used to get single tree predictions, or averages of diffferent trees
         with different weightings. Duplicate entries are allowed, so if ``trees = [0,1,1]``
         this will predict the weighted average prediction of only trees 0 and 1 weighted by::

            predict(..., trees = [0,1,1]) = (predict(..., trees = [0]) +
                                            2*predict(..., trees = [1])) / 3

         note we must have ``exact = True``, and ``aggregation = "average"`` to use tree indices. Defaults to using
         all trees equally weighted.
        :type trees: *array_like, optional*
        :param weightMatrix: An indicator of whether or not we should also return a
         matrix of the weights given to each training observation when making each
         prediction. When getting the weight matrix, aggregation must be one of
         'average', 'oob', and 'doubleOOB'. his is a normal text paragraph.
        :type weightMatrix: *bool, optional, default=False*
        :return: An array of predicted responses.
        :rtype: numpy.array

        """

        # Preprocess the data. We only run the data checker if ridge is turned on,
        # because even in the case where there were no NAs in train, we still want to predict.

        preprocessing.forest_checker(self)

        if newdata is not None:
            if not (isinstance(newdata, (pd.DataFrame, pd.Series, list)) or type(newdata).__module__ == np.__name__):
                raise AttributeError(
                    "newdata must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list"
                )

            newdata = (pd.DataFrame(newdata)).copy()
            newdata.reset_index(drop=True, inplace=True)
            preprocessing.testing_data_checker(self, newdata)

            if self.processed_dta.feat_names is not None:
                if not all(newdata.columns == self.processed_dta.feat_names):
                    warnings.warn("newdata columns have been reordered so that they match the training feature matrix")
                    newdata = newdata[self.processed_dta.feat_names]

        if trees is not None:
            if not preprocessing.predict_exact(newdata, exact) or aggregation != "average":
                raise ValueError("When using tree indices, we must have exact = True and aggregation = 'average' ")

            if any((not isinstance(i, (int, np.integer))) or (i < -self.ntree) or (i >= self.ntree) for i in trees):
                raise ValueError("trees must contain indices which are integers between -ntree and ntree-1")

        if aggregation == "oob":
            predictions, weight_matrix = self._aggregation_oob(newdata, exact, return_weight_matrix)

        elif aggregation == "doubleOOB":
            predictions, weight_matrix = self._aggregation_double_oob(newdata, exact, return_weight_matrix)

        elif aggregation == "coefs":
            predictions, weight_matrix, coefficients = self._aggregation_coefs(
                newdata, exact, self._get_seed(seed), self._get_nthread(nthread)
            )
            return {
                "predictions": predictions,
                "coef": np.lib.stride_tricks.as_strided(
                    coefficients,
                    shape=(
                        self.processed_dta.n_observations,
                        self.processed_dta.linear_feature_cols.size + 1,
                    ),
                    strides=(
                        coefficients.itemsize * (self.processed_dta.linear_feature_cols.size + 1),
                        coefficients.itemsize,
                    ),
                ),
            }

        else:
            predictions, weight_matrix, _ = self._aggregation_fallback(
                newdata,
                aggregation,
                exact,
                self._get_seed(seed),
                self._get_nthread(nthread),
                return_weight_matrix,
                trees,
            )

        if return_weight_matrix:
            return {
                "predictions": predictions,
                "weightMatrix": np.lib.stride_tricks.as_strided(
                    weight_matrix,
                    shape=(self._get_n_preds(newdata), self.processed_dta.n_observations),
                    strides=(
                        weight_matrix.itemsize * self.processed_dta.n_observations,
                        weight_matrix.itemsize,
                    ),
                ),
            }

        return predictions

    def get_oob(self, no_warning: bool = False) -> Optional[float]:
        """
        Calculate the out-of-bag error of a given forest. This is done
        by using the out-of-bag predictions for each observation, and calculating the
        MSE over the entire forest.

        :param noWarning: A flag to not display warnings.
        :type noWarning: *bool, optional, default=False*
        :return: The OOB error of the forest.
        :rtype: float

        """

        preprocessing.forest_checker(self)
        if (not self.replace) and (self.ntree * (self.processed_dta.n_observations - self.sampsize)) < 10:
            if not no_warning:
                warnings.warn("Samples are drawn without replacement and sample size is too big!")
            return None

        preds = self.predict(newdata=None, aggregation="oob", exact=True)
        preds = preds[~np.isnan(preds)]

        # Only calc mse on non missing predictions
        y_true = self.processed_dta["y"]
        y_true = y_true[~np.isnan(y_true)]

        if self.scale:
            y_true = y_true * self.processed_dta.col_sd[-1] + self.processed_dta.col_means[-1]

        return np.mean((y_true - preds) ** 2)

    # TODO: CHANGE get_vi after weightmatrix!!!!!!!!!!!!!!!!!!!#######
    # TODO: Check out Bottleneck for faster numpy!!!!!!!!!!!!!!!!!!!!!!!!!
    def get_vi(self, no_warning: bool = False) -> Optional[np.ndarray]:
        """
        Calculate the percentage increase in OOB error of the forest
        when each feature is shuffled.

        :param noWarning: A flag to not display warnings.
        :type noWarning: *bool, optional, default=False*
        :return: The variable importance of the forest.
        :rtype: numpy.array

        """

        preprocessing.forest_checker(self)
        if (not self.replace) and (self.ntree * (self.processed_dta.n_observations - self.sampsize)) < 10:
            if not no_warning:
                warnings.warn("Samples are drawn without replacement and sample size is too big!")
            return None

        cpp_vi = extension.get_vi(self.forest)

        result = np.empty(self.processed_dta.num_columns)
        for i in range(self.processed_dta.num_columns):
            result[i] = extension.vector_get(cpp_vi, i)

        return result

    def get_ci(
        self,
        newdata: Union[pd.DataFrame, pd.Series, np.ndarray],
        level: float = 0.95,
        n_bootstrap_draws: int = 100,
        method: str = "OOB-conformal",
    ) -> dict:

        """
        For a new set of features, calculate the confidence intervals for each new observation.

        :param newdata: A set of new observations for which we want to predict the
         outcomes and use confidence intervals.
        :type newdata: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols]*
        :param level: The confidence level at which we want to make our intervals.
        :type level: *double, optional, default=.95*
        :param B: Number of bootstrap draws to use when using ``method = "OOB-bootstrap"``
        :type B: *int, optional, default=100*
        :param method: A flag for the different ways to create the confidence intervals.
         Right now we have two ways of doing this. One is the 'OOB-bootstrap' flag which
         uses many bootstrap pulls from the set of OOB trees then with these different
         pulls, we use the set of trees to predict for the new feature and give the
         confidence set over the many bootstrap draws. The other method - 'OOB-conformal' -
         creates intervals by taking the set of doubleOOB trees for each observation, and
         using the predictions of these trees to give conformal intervals. So for an
         observation obs_i, let S_i be the set of trees for which obs_i was in neither
         the splitting set nor the averaging set (or the set of trees for which obs_i
         was "doubleOOB"), we then predict for obs_i with only the trees in S_i::

            doubleOOB_tree_preds = S_i.predict(obs_i)
            CI(obs_i, level = .95) = quantile(doubleOOB_tree_preds - y_i, [0.025, 0.975])

         The 'local-conformal' option takes the residuals of each training point (using)
         OOB predictions, and then uses the weights of the random forest to determine
         the quantiles of the residuals in the local neighborhood of the predicted point.
        :type method: *str, optional, default='OOB-conformal'*
        :param noWarning: A flag to not display warnings.
        :type noWarning: *bool, optional, default=False*
        :return: The confidence intervals for each observation in newdata.
        :rtype: dict

        """

        if method not in ["OOB-conformal", "OOB-bootstrap", "local-conformal"]:
            raise ValueError("Method must be one of OOB-conformal, OOB-bootstrap, or local-conformal")

        if method == "OOB-conformal" and not (self.oob_honest and self.double_bootstrap):
            raise ValueError("We cannot do OOB-conformal intervals unless both OOBhonest and doubleBootstrap are True")

        if method == "OOB-bootstrap" and not self.oob_honest:
            raise ValueError("We cannot do OOB-bootstrap intervals unless OOBhonest is True")

        if method == "local-conformal" and not self.oob_honest:
            raise ValueError("We cannot do local-conformal intervals unless OOBhonest is True")

        if method == "OOB-bootstrap":
            # Now we do B bootstrap pulls of the trees in order to do prediction
            # intervals for newdata
            prediction_array = pd.DataFrame(np.empty((len(newdata.index), n_bootstrap_draws)))

            for i in range(n_bootstrap_draws):
                bootstrap = np.random.choice(self.ntree, size=self.ntree, replace=True, p=None)

                pred = self.predict(newdata=newdata, trees=bootstrap)
                prediction_array.iloc[:, i] = pred

            quantiles = np.quantile(prediction_array, [(1 - level) / 2, 1 - (1 - level) / 2], axis=1)

            return {
                "Predictions": self.predict(newdata=newdata),
                "CI.upper": quantiles[1, :],
                "CI.lower": quantiles[0, :],
                "level": level,
            }

        if method == "OOB-conformal":
            # Get double OOB predictions and the residuals
            y_pred = self.predict(aggregation="doubleOOB")

            if self.scale:
                res = y_pred - (self.processed_dta.y * self.processed_dta.col_sd[-1] + self.processed_dta.col_means[-1])
            else:
                res = y_pred - self.processed_dta.y

            # Get (1-level) / 2 and 1 - (1-level) / 2 quantiles of the residuals
            quantiles = np.quantile(res, [(1 - level) / 2, 1 - (1 - level) / 2])

            # Get predictions on newdata
            predictions = self.predict(newdata)

            return {
                "Predictions": predictions,
                "CI.upper": predictions + quantiles[1],
                "CI.lower": predictions + quantiles[0],
                "level": level,
            }

        # method == 'local-conformal'

        # oob_preds = self.predict(aggregation="oob")
        # if self.scale:
        #    oob_res = (
        #        self.processed_dta.y * self.processed_dta.col_sd[-1] + self.processed_dta.col_means[-1]
        #    ) - oob_preds
        # else:
        #    oob_res = self.processed_dta.y - oob_preds
        # TODO: weightmatrix not implemented yet!!!!

    def predict_info(self, aggregation: str = "oob") -> None:
        """
        Get the observations which are used to predict for a set of new
        observations using either all trees (for out of sample observations), or
        tree for which the observation is out of averaging set or out of sample entirely.

        :param newdata: Data on which we want to do predictions. Must be the same length
         as the training set if we are doing 'oob' or 'doubleOOB' aggregation.
        :type newdata: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols]*
        :param aggregation: Specifies which aggregation version is used to predict for the
         observation, must be one of 'average', 'oob', and 'doubleOOB'.
        :type aggregation: *str, optional, default='oob'*
        :return: A dictionary with four entries. 'weightMatrix' is a matrix specifying the
         weight given to training observation i when prediction on observation j.
         'avgIndices' gives the indices which are in the averaging set for each new
         observation. 'avgWeights' gives the weights corresponding to each averaging
         observation returned in 'avgIndices'. 'obsInfo' gives the full observation vectors
         which were used to predict for an observation, as well as the weight given
         each observation.
        :rtype: dict

        """

        if aggregation not in ["average", "oob", "doubleOOB"]:
            raise ValueError("Aggregation must be one of average, oob, or doubleOOB")

        # TODO: weightmatrix not implemented yet!!!!

    def decision_path(self, X: Union[pd.DataFrame, pd.Series, np.ndarray], tree_idx: int = 0) -> np.ndarray:
        """
        Gets the decision path in the forest.

        :param X: Testing samples. For each observation in X, we will get its decision path.
        :type X: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols]*
        :param tree_idx: The index of the tree in the forest where the path will be found.
        :type tree_idx: *int, optional, default=0*
        :return: A node indicator matrix, where each entry denotes the id of the corresponding
         node.
        :rtype: numpy.ndarray

        """

        X = pd.DataFrame(X)

        result = np.empty(len(X.index), dtype=object)
        for i in range(len(X.index)):
            obs = X.iloc[i, :].values
            path_ptr = extension.get_path(self.forest, obs, tree_idx)

            path_length = int(extension.vector_get_size_t(path_ptr, 0))
            path_array = np.empty(path_length, dtype=np.intc)

            for j in range(path_length):
                path_array[j] = int(extension.vector_get_size_t(path_ptr, j + 1))

            result[i] = path_array

        return result

    def score(
        self, X: Union[pd.DataFrame, pd.Series, List], y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Gets the coefficient of determination (R :sup:`2`).

        :param X: Testing samples.
        :type X: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols]*
        :param y: True outcome values of X.
        :type y: *array_like of shape [nsamples,]*
        :param sample_weight: Sample weights. Uses equal weights by default.
        :type sample_weight: *array_like of shape [nsamples,], optional, default=None*
        :return: The value of R :sup:`2`.
        :rtype: float

        """
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(newdata=X, aggregation="average"), sample_weight=sample_weight)

    def translate_tree(self, tree_ids: Optional[Union[int, np.ndarray]] = None) -> None:
        """
        Given a trained forest, translates the selected trees by allowing access to its undelying structure.
        After translating tree *i*, its structure will be stored as a dictionary in :ref:`Py_forest <translate-label>`
        and can be accessed by ``[RandomForest object].Py_forest[i]``. Check out the :ref:`Py_forest <translate-label>`
        attribute for more details about its structure.

        :param tree_ids: The indices of the trees to be translated. By default, all the trees in the forest
         are translated.
        :type tree_ids: *int/array_like, optional*
        :rtype: None
        """

        if len(self.py_forest) == 0:
            self.py_forest = [{} for _ in range(self.ntree)]

        if tree_ids is None:
            idx = np.arange(self.ntree)
        else:
            if isinstance(tree_ids, (int, np.integer)):
                idx = np.array([tree_ids])
            else:
                idx = np.array(tree_ids)

        for cur_id in idx:

            if self.py_forest[cur_id]:
                continue

            num_nodes = extension.get_tree_node_count(self.forest, cur_id)

            # Initialize arrays to pass to C
            split_info = np.empty(self.sampsize + 1, dtype=np.intc)
            averaging_info = split_info

            tree_info = np.empty(num_nodes * 8 + 1, dtype=np.double)

            extension.fill_tree_info(self.forest, cur_id, tree_info, split_info, averaging_info)

            self.py_forest[cur_id]["children_right"] = np.empty(num_nodes, dtype=np.intc)
            self.py_forest[cur_id]["children_left"] = np.empty(num_nodes, dtype=np.intc)
            self.py_forest[cur_id]["feature"] = np.empty(num_nodes, dtype=np.intc)
            self.py_forest[cur_id]["n_node_samples"] = np.empty(num_nodes, dtype=np.intc)
            self.py_forest[cur_id]["threshold"] = np.empty(num_nodes, dtype=np.double)
            self.py_forest[cur_id]["values"] = np.empty(num_nodes, dtype=np.double)
            self.py_forest[cur_id]["na_left_count"] = np.empty(num_nodes, dtype=np.intc)
            self.py_forest[cur_id]["na_right_count"] = np.empty(num_nodes, dtype=np.intc)

            for i in range(num_nodes):
                self.py_forest[cur_id]["children_right"][i] = int(tree_info[i])
                self.py_forest[cur_id]["children_left"][i] = int(tree_info[num_nodes + i])
                self.py_forest[cur_id]["feature"][i] = int(tree_info[num_nodes * 2 + i])
                self.py_forest[cur_id]["n_node_samples"][i] = int(tree_info[num_nodes * 3 + i])
                self.py_forest[cur_id]["threshold"][i] = tree_info[num_nodes * 4 + i]
                self.py_forest[cur_id]["values"][i] = tree_info[num_nodes * 5 + i]
                self.py_forest[cur_id]["na_left_count"][i] = int(tree_info[num_nodes * 6 + i])
                self.py_forest[cur_id]["na_right_count"][i] = int(tree_info[num_nodes * 7 + i])

            num_split_idx = int(split_info[0])
            self.py_forest[cur_id]["splitting_sample_idx"] = np.empty(num_split_idx, dtype=np.intc)
            for i in range(num_split_idx):
                self.py_forest[cur_id]["splitting_sample_idx"][i] = int(split_info[i + 1])

            num_av_idx = int(averaging_info[0])
            self.py_forest[cur_id]["averaging_sample_idx"] = np.empty(num_av_idx, dtype=np.intc)
            for i in range(num_av_idx):
                self.py_forest[cur_id]["averaging_sample_idx"][i] = int(averaging_info[i + 1])

            self.py_forest[cur_id]["seed"] = int(tree_info[num_nodes * 8])

    def corrected_predict(
        self,
        newdata: Optional[Any] = None,
        feats: Optional[Any] = None,
        nrounds: int = 0,
        linear: bool = True,
        double: bool = False,
        simple: bool = True,
        verbose: bool = False,
        use_residuals: bool = False,
        adaptive: bool = False,
        monotone: bool = False,
        num_quants: int = 5,
        forestry_parameters: dict = {},
        keep_fits: bool = False,
    ) -> Union[np.ndarray, Dict]:
        """
        Perform predictions given the forest using a bias correction based on
        the out of bag predictions on the training set. By default, we use a final linear
        correction based on the leave-one-out hat matrix after doing 'nrounds' nonlinear
        corrections.

        :param newdata: Dataframe on which to predict. If this is left *None*, we
         predict on the in sample data.
        :type newdata: *pandas.DataFrame, pandas.Series, numpy.ndarray, 2d list of shape [nsamples, ncols],
         optional, default=None*
        :param feats: A list of feature indices which should be included in the bias
         correction. By default only the outcome and predicted outcomes are used.
        :type feats: *array_like, optional, default=None*
        :param nrounds: The number of nonlinear bias correction steps which should be
         taken. By default, just a single linear correction is used.
        :type nrounds: *int, optional, default=0*
        :param linear: A flag indicating whether or not we want to do a final linear
         bias correction after doing the nonlinear corrections.
        :type linear: *bool, optional, default=True*
        :param double: A flag indicating if one should use ``aggregation = "doubleOOB"`` for
         the initial predictions rather than ``aggregation = "oob"``.
        :type double: *bool, optional, default=False*
        :param simple: flag indicating whether we should do a simple linear adjustment
         or do different adjustments by quantiles.
        :type simple: *bool, optional, default=True*
        :param verbose: A flag which displays the bias of each qunatile.
        :type verbose: *bool, optional, default=False*
        :param use_residuals: A flag indicating if we should use the residuals to fit the
         bias correction steps. Defualt is *False*, which means that we will use *Y*
         rather than *Y-Y_hat* as the regression outcome in the bias correction steps.
        :type use_residuals: *bool, optional, default=False*
        :param adaptive: A flag to indicate whether we use *adaptiveForestry* or not in the
         regression step. *adaptiveForestry* is not implemented yet, so the default is *False*.
        :type adaptive: *bool, optional, default=False*
        :param monotone: A flag to indicate whether or not we should use monotonicity
         in the regression of *Y* on *Y_hat* (when doing forest correction steps).
         If *True*, will constrain the corrected prediction for *Y* to be monotone in the
         original prediction of *Y*.
        :type monotone: *bool, optional, default=False*
        :param num_quants: Number of quantiles to use when doing quantile specific bias
         correction. Will only be used if ``simple = False``.
        :type num_quants: *int, optional, default=5*
        :param params_forestry: A dictionary of parameters to pass to the subsequent *RandomForest*
         calls. Note that these forests will be trained on features of dimension
         ``len(feats) + 1`` as the correction forests are trained using the additional feature *Y_hat*,
         so monotonic constraints etc given to this list should be of size ``len(feats) + 1``.
         Defaults to the standard *RandomForest* parameters for any parameters that are
         not included in the dictionary.
        :type params_forestry: *dict, optional*
        :param keep_fits: A flag that indicates if we should save the intermediate
         forests used for the bias correction. If this is *True*, we return a list of
         the *RandomForest* objects for each iteration in the bias correction.
        :type keep_fits: *bool, optional, default=False*
        :return: An array of the bias corrected predictions
        :rtype: numpy.array
        """

        # To avoid false pd warnings
        pd.options.mode.chained_assignment = None

        # Check allowed settings for the bias correction
        if nrounds < 1 and not linear:
            raise ValueError(
                "We must do at least one round of bias corrections, with either linear = True or nrounds > 0."
            )

        if nrounds < 0 or not isinstance(nrounds, int):
            raise ValueError("nrounds must be a non negative integer.")

        if feats is not None:
            if any(
                not isinstance(x, (int, np.integer))
                or x < -self.processed_dta.num_columns  # pylint: disable=invalid-unary-operand-type
                or x >= self.processed_dta.num_columns
                for x in feats
            ):
                raise ValueError("feats must be  a integer between -ncol and ncol(x)-1")

        # Check the parameters match parameters for RandomForest or adaptiveForestry
        if not adaptive:
            forestry_args = set(self.get_parameters().keys())

        for parameter in forestry_parameters:
            if parameter not in forestry_args:
                raise ValueError(f"Invalid parameter in params.forestry: {parameter}")

        if double:
            aggregation = "doubleOOB"
        else:
            aggregation = "oob"

        # First get out of bag preds
        oob_preds = self.predict(aggregation=aggregation)

        if feats is None:
            adjust_data = pd.DataFrame({"Y": self.processed_dta["y"], "Y.hat": oob_preds})
        else:
            adjust_data = self.processed_dta["processed_x"].iloc[:, feats]
            adjust_data.columns = ["V" + str(x) for x in range(len(feats))]
            adjust_data["Y"] = self.processed_dta["y"]
            adjust_data["Y.hat"] = oob_preds

        # Store the RF fits
        rf_fits = []

        if nrounds > 0:
            for _ in range(nrounds):
                # Set right outcome to regress for regression step
                if use_residuals:
                    y_reg = adjust_data["Y"] - adjust_data["Y.hat"]
                else:
                    y_reg = adjust_data["Y"]

                if monotone:
                    # Set default params for monotonicity in the Y.hat feature
                    parameters = forestry_parameters.copy()
                    parameters["OOBhonest"] = True
                    parameters["monotoneAvg"] = True
                    monotone_constraits = np.zeros(len(adjust_data.columns) - 1)
                    monotone_constraits[-1] = 1

                    forest = RandomForest(**parameters)
                    forest.fit(
                        x=adjust_data.loc[:, adjust_data.columns != "Y"],
                        y=y_reg,
                        monotonic_constraints=monotone_constraits,
                    )

                else:
                    # Set default RandomForest params
                    parameters = forestry_parameters.copy()
                    parameters["OOBhonest"] = True

                    forest = RandomForest(**parameters)
                    forest.fit(x=adjust_data.loc[:, adjust_data.columns != "Y"], y=y_reg)

                pred = forest.predict(adjust_data.loc[:, adjust_data.columns != "Y"], aggregation=aggregation)

                # Stror the ith fit
                rf_fits.append(forest)

                # If we predicted some residuals, we now have to add the old Y.hat to them
                # to get the new Y.hat
                if use_residuals:
                    pred += adjust_data["Y.hat"]

                # Adjust the predicted Y hats
                adjust_data["Y.hat"] = pred

            # if we have a new feature, we need to run the correction fits on that as well
            if newdata is not None:
                # Get initial predictions
                if feats is None:
                    pred_data = pd.DataFrame({"Y.hat": self.predict(newdata=newdata)})
                else:
                    pred_data = pd.DataFrame(newdata.iloc[:, feats])
                    pred_data["Y.hat"] = self.predict(newdata=newdata)  # aggregation = agg ?????

                # Set column names to follow a format matching the features used
                if feats is not None:
                    pred_data.columns = ["V" + str(x) for x in range(len(feats))] + ["Y.hat"]

                for i in range(nrounds):
                    adjusted_pred = rf_fits[i].predict(newdata=pred_data)  # aggregation = agg ?????
                    pred_data["Y.hat"] = adjusted_pred  # doesnt make sense??????

        if newdata is not None:
            if nrounds > 0:
                preds_initial = pred_data["Y.hat"]
            else:
                preds_initial = self.predict(newdata=newdata)

        # Given a dataframe with Y and Y.hat at least, fits an OLS and gives the LOO
        # predictions on the sample
        def loo_pred_helper(data_frame: pd.DataFrame) -> dict:

            Y = data_frame["Y"]
            X = data_frame.loc[:, data_frame.columns != "Y"]
            X = sm.add_constant(X)

            adjust_lm = sm.OLS(Y, X).fit()

            cv = LeaveOneOut()
            cv_pred = np.empty(Y.size)

            for i, (train, test) in enumerate(cv.split(X)):
                # split data
                X_train, X_test = X.iloc[train, :], X.iloc[test, :]
                y_train, _ = Y[train], Y[test]

                # fit model
                model = sm.OLS(y_train, X_train).fit()
                cv_pred[i] = model.predict(X_test)

            return {"insample_preds": cv_pred, "adjustment_model": adjust_lm}

        if linear:
            # Now do linear adjustment
            if simple:
                # Now we either return the adjusted in sample predictions, or the
                # out of sample predictions scaled according to the adjustment model
                if newdata is None:
                    preds_adjusted = loo_pred_helper(adjust_data)["insample_preds"]
                else:
                    model = loo_pred_helper(adjust_data)["adjustment_model"]

                    if feats is None:
                        data_pred = pd.DataFrame({"Y.hat": preds_initial})
                    else:
                        data_pred = pd.DataFrame(newdata.iloc[:, feats])
                        data_pred.columns = ["V" + str(x) for x in range(len(feats))]
                        data_pred["Y.hat"] = preds_initial

                    preds_adjusted = np.array(model.predict(sm.add_constant(data_pred)))

            # Not simple
            else:
                # split Yhat into quantiles
                Y_hat = adjust_data["Y.hat"]
                Y = np.array(adjust_data["Y"])

                cuts, training_quantiles = pd.qcut(Y_hat, num_quants, retbins=True)
                q_idx = np.array(cuts.cat.codes)

                new_pred = np.empty(Y_hat.size)

                for i in range(num_quants):
                    # split data
                    mask = q_idx == i
                    if verbose:
                        bias = np.mean(Y[mask] - Y_hat[mask])
                        print(f"Quantile {i}: {cuts.cat.categories[i]}, Bias: {bias}")
                    # Again we have to check if the data was in or out of sample, and do predictions
                    # accordingly --????? why aren't we doing this?
                    new_pred[mask] = loo_pred_helper(adjust_data.loc[mask, :].reset_index(drop=True))["insample_preds"]

                if newdata is not None:
                    # Now we have fit the Q_num different models, we take the models and use
                    # split Yhat into quantiles
                    np.put(training_quantiles, -1, np.inf)
                    np.put(training_quantiles, 0, -np.inf)

                    # Get the quantile each testing observation falls into
                    testing_quantiles = np.empty(preds_initial.size, dtype=np.intc)
                    for i in range(preds_initial.size):  # ?????? Why use two loops??
                        testing_quantiles[i] = np.argmax(training_quantiles >= preds_initial[i]) - 1

                    # Now predict for each index set using the right model
                    preds_adjusted = np.empty(len(newdata.index))
                    for i in range(training_quantiles.size - 1):
                        mask = testing_quantiles == i

                        if feats is None:  # ???? Check this later
                            pred_df = pd.DataFrame({"Y.hat": preds_initial[mask].reset_index(drop=True)})
                        else:
                            pred_df = pd.DataFrame(newdata.iloc[mask, feats].reset_index(drop=True))
                            pred_df.columns = ["V" + str(x) for x in range(len(feats))]
                            pred_df["Y.hat"] = preds_initial[mask].reset_index(drop=True)

                        fit_i = sm.OLS(
                            Y[q_idx == i],
                            sm.add_constant(
                                adjust_data.loc[q_idx == i, adjust_data.columns != "Y"].reset_index(drop=True)
                            ),
                        ).fit()
                        preds_adjusted[mask] = fit_i.predict(sm.add_constant(pred_df))

                else:
                    preds_adjusted = new_pred

            if not keep_fits:
                return preds_adjusted

            return {"predictions": preds_adjusted, "fits": rf_fits}

        # Not linear
        else:
            if not keep_fits:
                return np.array(adjust_data.iloc[:, -1])
            return {
                "predictions": np.array(adjust_data.iloc[:, -1]),
                "fits": rf_fits,
            }

    def get_split_proportions(self) -> np.ndarray:
        """
        Retrieves the proportion of splits for each feature in the given
        *RandomForest* object. These proportions are calculated as the number of splits
        on feature *i* in the entire forest over total the number of splits in the forest.

        :return: An array of length equal to the number of columns
        :rtype: numpy.array
        """

        split_nums = np.zeros(self.processed_dta.num_columns)

        self.translate_tree()
        for i in range(self.ntree):
            for feature in self.py_forest[i]["feature"]:
                if feature >= 0:
                    split_nums[feature] += 1

        return split_nums / np.sum(split_nums)

    def get_parameters(self) -> dict:
        """
        Get the parameters of `RandomForest`.

        :return: A dictionary mapping parameter names of the `RandomForest` to their values.
        :rtype: dict
        """

        return {
            parameter: value
            for parameter, value in self.__dict__.items()
            if parameter not in ["forest", "dataframe", "processed_dta", "py_forest", "__pydantic_initialised__"]
        }

    def set_parameters(self, **new_parameters: dict) -> Self:
        """
        Set the parameters of the *RandomForest*.

        :param **params: Forestry parameters.
        :type **params: *dict*
        :return: A new *RandomForest* object with the given parameters.
         Note: this reinitializes the *RandomForest* object,
         so fit must be called on the new estimator.
        :rtype: *RandomForest* -- MIGHT CHANGE THE NAME
        """

        if not new_parameters:
            return self

        current_parameters = self.get_parameters()
        for parameter in new_parameters:
            if parameter not in current_parameters.keys():
                raise ValueError(
                    f"Invalid parameter {parameter} for RandomForest. Check the list of available parameters "
                    "with `estimator.get_parameters().keys()`."
                )

        self.__init__(**{**current_parameters, **new_parameters})  # pylint: disable=unnecessary-dunder-call
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["dataframe"]
        del state["forest"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        group_vector = (
            pd.to_numeric(state["processed_dta"].groups)
            if state["processed_dta"].groups is not None
            else np.repeat(0, state["processed_dta"].n_observations)
        )
        state["dataframe"] = extension.get_data(
            np.ascontiguousarray(
                pd.concat(
                    [
                        state["processed_dta"].processed_x,
                        pd.Series(state["processed_dta"].y),
                    ],
                    axis=1,
                ).values[:, :],
                np.double,
            ).ravel(),
            state["processed_dta"].categorical_feature_cols,
            state["processed_dta"].categorical_feature_cols.size,
            state["processed_dta"].linear_feature_cols,
            state["processed_dta"].linear_feature_cols.size,
            state["processed_dta"].feature_weights,
            state["processed_dta"].feature_weights_variables,
            state["processed_dta"].feature_weights_variables.size,
            state["processed_dta"].deep_feature_weights,
            state["processed_dta"].deep_feature_weights_variables,
            state["processed_dta"].deep_feature_weights_variables.size,
            state["processed_dta"].observation_weights,
            state["processed_dta"].monotonic_constraints,
            group_vector,
            state["monotone_avg"],
            state["processed_dta"].symmetric,
            state["processed_dta"].symmetric.size,
            state["processed_dta"].n_observations,
            state["processed_dta"].num_columns + 1,
            state["seed"],
        )

        tree_info = np.empty(state["ntree"] * 3, dtype=np.intc)
        total_nodes, total_split_idx, total_av_idx = 0, 0, 0
        for i in range(state["ntree"]):
            tree_info[3 * i] = state["py_forest"][i]["children_right"].size
            total_nodes += tree_info[3 * i]

            tree_info[3 * i + 1] = state["py_forest"][i]["splitting_sample_idx"].size
            total_split_idx += tree_info[3 * i + 1]

            tree_info[3 * i + 2] = state["py_forest"][i]["averaging_sample_idx"].size
            total_av_idx += tree_info[3 * i + 2]

        thresholds = np.empty(total_nodes, dtype=np.double)
        features = np.empty(total_nodes, dtype=np.intc)
        na_left_counts = np.empty(total_nodes, dtype=np.intc)
        na_right_counts = np.empty(total_nodes, dtype=np.intc)
        sample_split_idx = np.empty(total_split_idx, dtype=np.intc)
        sample_av_idx = np.empty(total_av_idx, dtype=np.intc)
        predict_weights = np.empty(total_nodes, dtype=np.double)
        tree_seeds = np.empty(state["ntree"], dtype=np.uintc)

        ind, ind_s, ind_a = 0, 0, 0
        for i in range(state["ntree"]):
            for j in range(tree_info[3 * i]):
                thresholds[ind] = state["py_forest"][i]["threshold"][j]
                features[ind] = state["py_forest"][i]["feature"][j]
                na_left_counts[ind] = state["py_forest"][i]["na_left_count"][j]
                na_right_counts[ind] = state["py_forest"][i]["na_right_count"][j]
                predict_weights[ind] = state["py_forest"][i]["values"][j]

                ind += 1

            for j in range(tree_info[3 * i + 1]):
                sample_split_idx[ind_s] = state["py_forest"][i]["splitting_sample_idx"][j]
                ind_s += 1

            for j in range(tree_info[3 * i + 2]):
                sample_av_idx[ind_a] = state["py_forest"][i]["averaging_sample_idx"][j]
                ind_a += 1

            tree_seeds[i] = state["py_forest"][i]["seed"]

        state["forest"] = extension.reconstruct_tree(
            state["dataframe"],
            state["ntree"],
            state["replace"],
            state["sampsize"],
            state["splitratio"],
            state["oob_honest"],
            state["double_bootstrap"],
            state["mtry"],
            state["nodesize_spl"],
            state["nodesize_avg"],
            state["nodesize_strict_spl"],
            state["nodesize_strict_avg"],
            state["min_split_gain"],
            state["max_depth"],
            state["interaction_depth"],
            state["seed"],
            state["nthread"],
            state["verbose"],
            state["middle_split"],
            state["max_obs"],
            state["min_trees_per_group"],
            state["processed_dta"].has_nas,
            state["linear"],
            not np.any(state["processed_dta"].symmetric),
            state["overfit_penalty"],
            state["double_tree"],
            tree_info,
            thresholds,
            features,
            na_left_counts,
            na_right_counts,
            sample_split_idx,
            sample_av_idx,
            predict_weights,
            tree_seeds,
        )

    # Saving and loading
    def save_forestry(self, filename: Path) -> None:
        self.translate_tree()

        with open(filename, "wb") as output_file:  # Overwrites any existing file.
            pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_forestry(filename: Path) -> Self:
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)  # nosec B301

    def __del__(self):
        # Free the pointers to foretsry and dataframe
        extension.delete_forestry(self.forest, self.dataframe)


# make linFeats same as symmetric...

# ????????????? min nodes strict or not in plotting - get_min_samples_leaf
# https://github.com/parrt/dtreeviz/blob/master/dtreeviz/models/shadow_decision_tree.py
