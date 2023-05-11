import os
import warnings
from math import ceil
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import check_X_y

from . import negative_float, negative_integer, positive_float, positive_integer
from .base_validator import BaseValidator


class FitValidator(BaseValidator):
    DEFAULT_MTRY: Final = None
    DEFAULT_SAMPSIZE: Final = None
    DEFAULT_MAX_OBS: Final = None
    DEFAULT_MAX_DEPTH: Final = None
    DEFAULT_INTERACTION_DEPTH: Final = None
    DEFAULT_DOUBLE_BOOTSTRAP: Final = None
    DEFAULT_DOUBLE_TREE: Final = False
    DEFAULT_SPLITRATIO: Final = 1.0
    DEFAULT_REPLACE: Final = True

    def validate_monotonic_constraints(self, forest, x, **kwargs):
        _, ncols = x.shape

        if "monotonic_constraints" not in kwargs:
            monotonic_constraints = np.zeros(ncols, dtype=np.intc)
        else:
            monotonic_constraints = np.array(kwargs["monotonic_constraints"], dtype=np.intc)

        if monotonic_constraints.size != ncols:
            raise ValueError("monotonic_constraints must have the size of x")
        if any(i not in (0, 1, -1) for i in monotonic_constraints):
            raise ValueError("monotonic_constraints must be either 1, 0, or -1")
        if any(i != 0 for i in monotonic_constraints) and forest.linear:
            raise ValueError("Cannot use linear splitting with monotonic_constraints")

        return monotonic_constraints

    def validate_observation_weights(self, forest, x, **kwargs):
        nrows, _ = x.shape

        if not self.validate_replace(forest, **kwargs):
            observation_weights = np.zeros(nrows, dtype=np.double)
        elif "observation_weights" not in kwargs:
            observation_weights = np.repeat(1.0, nrows)
        else:
            observation_weights = np.array(kwargs["observation_weights"], dtype=np.double)

        if observation_weights.size != nrows:
            raise ValueError("observation_weights must have length len(x)")
        if any(i < 0 for i in observation_weights):
            raise ValueError("The entries in observation_weights must be non negative")
        if self.validate_replace(forest, **kwargs) and np.sum(observation_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in observation_weights")

        return observation_weights

    def validate_lin_feats(self, x, **kwargs):
        _, ncols = x.shape

        if "lin_feats" not in kwargs:
            lin_feats = np.arange(ncols, dtype=np.ulonglong)
        else:
            lin_feats = pd.unique(np.array(kwargs["lin_feats"], dtype=np.ulonglong))

        if any(i < 0 or i >= ncols for i in lin_feats):
            raise ValueError("lin_feats must contain positive integers less than len(x.columns).")

        return lin_feats

    def validate_feature_weights(self, x, **kwargs):
        _, ncols = x.shape

        if "feature_weights" not in kwargs:
            feature_weights = np.repeat(1.0, ncols)
            interaction_variables = [] if "interaction_variables" not in kwargs else kwargs["interaction_variables"]
            feature_weights[interaction_variables] = 0.0
        else:
            feature_weights = np.array(kwargs["feature_weights"], dtype=np.double)

        if feature_weights.size != ncols:
            raise ValueError("feature_weights must have length len(x.columns)")

        if any(i < 0 for i in feature_weights):
            raise ValueError("The entries in feature_weights must be non negative")

        if np.sum(feature_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in feature_weights")

        return feature_weights

    def validate_deep_feature_weights(self, x, **kwargs):
        _, ncols = x.shape

        if "deep_feature_weights" not in kwargs:
            deep_feature_weights = np.repeat(1.0, ncols)
        else:
            deep_feature_weights = np.array(kwargs["deep_feature_weights"], dtype=np.double)

        if deep_feature_weights.size != ncols:
            raise ValueError("deep_feature_weights must have length len(x.columns)")

        if any(i < 0 for i in deep_feature_weights):
            raise ValueError("The entries in deep_feature_weights must be non negative")

        if np.sum(deep_feature_weights) == 0:
            raise ValueError("There must be at least one non-zero weight in deep_feature_weights")

        return deep_feature_weights

    def validate_groups(self, **kwargs):
        if "groups" in kwargs:
            groups = kwargs["groups"]
            if len(groups.unique()) == 1:
                raise ValueError("groups must have more than 1 level to be left out from sampling.")
            return groups
        return None

    def validate_mtry(self, x, **kwargs) -> int:
        _, ncols = x.shape

        if "mtry" in kwargs:
            mtry = kwargs["mtry"]
        else:
            mtry = __class__.DEFAULT_MTRY

        if mtry is None:
            mtry = max((ncols // 3), 1)

        if mtry > ncols:
            raise ValueError("mtry cannot exceed total amount of features in x.")

        if not positive_integer(mtry):
            raise ValueError("mtry must be positive_integer")
        return mtry

    def validate_sampsize(self, forest, x, **kwargs) -> int:
        nrows, _ = x.shape

        if "sampsize" in kwargs:
            sampsize = kwargs["sampsize"]
        else:
            sampsize = __class__.DEFAULT_SAMPSIZE

        if sampsize is None:
            sampsize = nrows if self.validate_replace(forest, **kwargs) else ceil(0.632 * nrows)

        # only if sample.fraction is given, update sampsize
        if forest.sample_fraction is not None:
            sampsize = ceil(forest.sample_fraction * nrows)

        if not self.validate_replace(forest, **kwargs) and sampsize > nrows:
            raise ValueError("You cannot sample without replacement with size more than total number of observations.")

        if not positive_integer(sampsize):
            raise ValueError("sampsize must be positive integer")
        return sampsize

    def validate_double_bootstrap(self, forest, **kwargs) -> bool:
        if "double_bootstrap" in kwargs:
            double_bootstrap = kwargs["double_bootstrap"]
        else:
            double_bootstrap = __class__.DEFAULT_DOUBLE_BOOTSTRAP

        if double_bootstrap is None:
            return forest.oob_honest

        return double_bootstrap

    def validate_replace(self, forest, **kwargs) -> bool:
        if "replace" in kwargs:
            replace = kwargs["replace"]
        else:
            replace = __class__.DEFAULT_REPLACE

        if forest.oob_honest and not replace:
            warnings.warn("replace must be set to TRUE to use OOBhonesty, setting this to True now")
            return True

        return replace

    def validate_double_tree(self, forest, **kwargs) -> bool:
        if "double_tree" in kwargs:
            double_tree = kwargs["double_tree"]
        else:
            double_tree = __class__.DEFAULT_DOUBLE_TREE

        if double_tree and self.validate_splitratio(forest, **kwargs) in (0, 1):
            warnings.warn("Trees cannot be doubled if splitratio is 1. We have set double_tree to False.")
            return False

        return double_tree

    def validate_splitratio(self, forest, **kwargs) -> float:
        if "splitratio" in kwargs:
            splitratio = kwargs["splitratio"]
        else:
            splitratio = __class__.DEFAULT_SPLITRATIO

        if forest.oob_honest and splitratio != 1:
            warnings.warn("oob_honest is set to true, so we will run OOBhonesty rather than standard honesty.")
            return 1

        if splitratio < 0 or splitratio > 1:
            raise ValueError("SplitRatio needs to be in range (0,1)")

        return splitratio

    def validate_max_obs(self, y, **kwargs):
        if "max_obs" in kwargs:
            max_obs = kwargs["max_obs"]
        else:
            max_obs = __class__.DEFAULT_MAX_OBS

        if max_obs is None:
            max_obs = y.size

        if not positive_integer(max_obs):
            raise ValueError("max_obs must be positive_integer")
        return max_obs

    def validate_max_depth(self, x, **kwargs):
        nrows, _ = x.shape

        if "max_depth" in kwargs:
            max_depth = kwargs["max_depth"]
        else:
            max_depth = __class__.DEFAULT_MAX_DEPTH

        if max_depth is None:
            max_depth = round(nrows / 2) + 1

        if not positive_integer(max_depth):
            raise ValueError("max_depth must be positive_integer")
        return max_depth

    def validate_interaction_depth(self, **kwargs):
        if "interaction_depth" in kwargs:
            interaction_depth = kwargs["interaction_depth"]
        else:
            interaction_depth = __class__.DEFAULT_INTERACTION_DEPTH

        if interaction_depth is None:
            interaction_depth = kwargs["max_depth"]
        else:
            if interaction_depth > kwargs["max_depth"]:
                warnings.warn(
                    "interaction_depth cannot be greater than max_depth. We have set interaction_depth to max_depth."
                )
                interaction_depth = kwargs["max_depth"]

        if not positive_integer(interaction_depth):
            raise ValueError("interaction_depth must be positive_integer")
        return interaction_depth

    def prevalidate(self, forest) -> None:
        for parameter in [
            "ntree",
            "nodesize_spl",
            "nodesize_avg",
            "nodesize_strict_spl",
            "nodesize_strict_avg",
            "fold_size",
        ]:
            if not positive_integer(getattr(forest, parameter)):
                raise ValueError(f"{parameter} must be positive integer")

        for parameter in ["seed", "nthread", "min_trees_per_fold"]:
            if negative_integer(getattr(forest, parameter)):
                raise ValueError(f"{parameter} must be non negative integer")

        if forest.sample_fraction and (
            not (positive_integer(forest.sample_fraction) or positive_float(forest.sample_fraction))
        ):
            raise ValueError("sample_fraction must be positive or None")

        if negative_float(forest.min_split_gain):
            raise ValueError("min_split_gain must be non negative float")

        if forest.nthread > os.cpu_count():
            raise ValueError("nthread cannot exceed total cores in the computer: " + str(os.cpu_count()))

        if forest.min_split_gain > 0 and not self.linear:
            raise ValueError("min_split_gain cannot be set without setting linear to be true.")

    def __call__(self, forest, x, y, *args, **kwargs):
        _, y = check_X_y(x, y, accept_sparse=False, force_all_finite="allow-nan")
        x = pd.DataFrame(x).copy()

        self.prevalidate(forest)

        if len(args) > 0:
            raise ValueError("There can be only 2 non-keyword arguments: X, y")

        # Check if the input dimension of x matches y
        nrows, _ = x.shape
        if nrows != y.size:
            raise ValueError("The dimension of input dataset x doesn't match the output y.")

        if np.isnan(y).any():
            raise ValueError("y contains missing data.")

        if len(x.columns[x.isnull().all()]) > 0:
            raise ValueError("Training data column cannot be all missing values.")

        if forest.linear and x.isnull().values.any():
            raise ValueError("Cannot do imputation splitting with linear.")

        kwargs["mtry"] = self.validate_mtry(x, **kwargs)
        kwargs["splitratio"] = self.validate_splitratio(forest, **kwargs)
        kwargs["replace"] = self.validate_replace(forest, **kwargs)
        kwargs["double_tree"] = self.validate_double_tree(forest, **kwargs)
        kwargs["sampsize"] = self.validate_sampsize(forest, x, **kwargs)
        kwargs["double_bootstrap"] = self.validate_double_bootstrap(forest, **kwargs)
        kwargs["max_obs"] = self.validate_max_obs(y, **kwargs)
        kwargs["max_depth"] = self.validate_max_depth(x, **kwargs)
        kwargs["interaction_depth"] = self.validate_interaction_depth(**kwargs)
        kwargs["monotonic_constraints"] = self.validate_monotonic_constraints(forest, x, **kwargs)
        kwargs["lin_feats"] = self.validate_lin_feats(x, **kwargs)
        kwargs["feature_weights"] = self.validate_feature_weights(x, **kwargs)
        kwargs["deep_feature_weights"] = self.validate_deep_feature_weights(x, **kwargs)
        kwargs["observation_weights"] = self.validate_observation_weights(forest, x, **kwargs)
        kwargs["groups"] = self.validate_groups(**kwargs)

        return self.function(forest, x, y, **kwargs)
