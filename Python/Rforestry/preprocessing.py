"""
LIBRARIES NEEDED
----------------------
"""

import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def has_nas(x: pd.DataFrame) -> bool:
    return x.isnull().values.any()


def get_default_lin_feats(x: pd.DataFrame, lin_feats: Optional[np.ndarray]) -> np.ndarray:
    _, ncol = x.shape
    if lin_feats is None:
        lin_feats = np.arange(ncol, dtype=np.ulonglong)
    else:
        lin_feats = np.array(lin_feats, dtype=np.ulonglong)
    return pd.unique(lin_feats)


def get_feat_names(x: Union[pd.DataFrame, pd.Series, List]) -> Optional[np.ndarray]:
    if isinstance(x, pd.DataFrame):
        return x.columns.values
    if type(x).__module__ == np.__name__ or isinstance(x, (list, pd.Series)):
        print(
            "x does not have column names. ",
            "The check that columns are provided in the same order when training and predicting will be skipped",
            file=sys.stderr,
        )
        return None

    raise AttributeError("x must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list")


def predict_exact(newdata: Optional[pd.DataFrame], exact: Optional[bool]) -> bool:
    if exact is not None:
        return exact
    if newdata is None:
        return True
    if len(newdata.index) > 1e5:
        return False
    return True


def find_match(A, B):
    """
    --------------------------------------

    Helper Function
    @return a nunpy array indicating the indices of first occurunces of
      the elements of A in B
    """

    temp_dict = {}

    for i in range(len(B)):
        if not str(B[i]) in temp_dict:
            temp_dict[str(B[i])] = i

    return np.array([temp_dict[str(val)] for val in A])


def forest_parameter_checker(
    ntree,
    replace,
    sampsize,
    sample_fraction,
    mtry,
    nodesizeSpl,
    nodesizeAvg,
    nodesizeStrictSpl,
    nodesizeStrictAvg,
    minSplitGain,
    maxDepth,
    interactionDepth,
    splitratio,
    OOBhonest,
    doubleBootstrap,
    seed,
    verbose,
    nthread,
    middleSplit,
    maxObs,
    linear,
    minTreesPerGroup,
    monotoneAvg,
    overfitPenalty,
    scale,
    doubleTree,
):
    """
    #-- Sanity Checker -------------------------------------------------------------
    #' @name forest_parameter_checker
    #' @description Check the input parameters to the RandomForest constructor
    #' @inheritParams RandomForest
    #' @return A tuple of parameters after checking the selected parameters are valid.
    """

    # Check for types and ranges

    if (not isinstance(ntree, (int, np.integer))) or ntree <= 0:
        raise ValueError("ntree must be a positive integer.")

    if not isinstance(replace, bool):
        raise AttributeError("replace must be True or False.")

    if (sampsize is not None) and ((not isinstance(sampsize, (int, np.integer))) or sampsize <= 0):
        raise ValueError("sampsize must be a positive integer.")

    if (sample_fraction is not None) and ((not isinstance(sample_fraction, (int, float))) or sample_fraction <= 0):
        raise ValueError("sample_fraction must be a positive integer.")

    if (mtry is not None) and ((not isinstance(mtry, (int, np.integer))) or mtry <= 0):
        raise ValueError("mtry must be a positive integer.")

    if (not isinstance(nodesizeSpl, (int, np.integer))) or nodesizeSpl <= 0:
        raise ValueError("nodesizeSpl must be a positive integer.")

    if (not isinstance(nodesizeAvg, (int, np.integer))) or nodesizeAvg <= 0:
        raise ValueError("nodesizeAvg must be a positive integer.")

    if (not isinstance(nodesizeStrictSpl, (int, np.integer))) or nodesizeStrictSpl <= 0:
        raise ValueError("nodesizeStrictSpl must be a positive integer.")

    if (not isinstance(nodesizeStrictAvg, (int, np.integer))) or nodesizeStrictAvg <= 0:
        raise ValueError("nodesizeStrictAvg must be a positive integer.")

    if (not isinstance(minSplitGain, (int, float))) or minSplitGain < 0:
        raise ValueError("minSplitGain must be a number greater than or equal to 0.")

    if (maxDepth is not None) and ((not isinstance(maxDepth, (int, np.integer))) or maxDepth <= 0):
        raise ValueError("maxDepth must be a positive integer.")

    if (interactionDepth is not None) and ((not isinstance(interactionDepth, int)) or interactionDepth <= 0):
        raise ValueError("interactionDepth must be a positive integer.")

    if (not isinstance(splitratio, (int, float))) or (splitratio < 0 or splitratio > 1):
        raise ValueError("splitratio must be a number between 0 and 1.")

    if not isinstance(OOBhonest, bool):
        raise AttributeError("OOBhonest must be True or False.")

    if not isinstance(doubleBootstrap, bool):
        raise AttributeError("doubleBootstrap must be True or False.")

    if (not isinstance(seed, int)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    if not isinstance(verbose, bool):
        raise AttributeError("verbose must be True or False.")

    if (not isinstance(nthread, int)) or nthread < 0:
        raise ValueError("nthread must be a nonnegative integer.")

    if nthread > 0:
        if nthread > os.cpu_count():
            raise ValueError("nthread cannot exceed total cores in the computer: " + str(os.cpu_count()))

    if not isinstance(middleSplit, bool):
        raise ValueError("middleSplit must be True or False.")

    if (maxObs is not None) and ((not isinstance(maxObs, (int, np.integer))) or maxObs <= 0):
        raise ValueError("maxObs must be a positive integer.")

    if not isinstance(linear, bool):
        raise ValueError("linear must be True or False.")

    if (minTreesPerGroup is not None) and (
        (not isinstance(minTreesPerGroup, (int, np.integer))) or minTreesPerGroup < 0
    ):
        raise ValueError("minTreesPerGroup must be a nonnegative integer.")

    if not isinstance(monotoneAvg, bool):
        raise AttributeError("monotoneAvg must be True or False.")

    if not isinstance(overfitPenalty, (int, float)):
        raise AttributeError("overfitPenalty must be a number.")

    if not isinstance(scale, bool):
        raise ValueError("scale must be True or False.")

    if not isinstance(doubleTree, bool):
        raise ValueError("doubleTree must be True or False.")

    # Some more checks

    if minSplitGain > 0 and (not linear):
        raise ValueError("minSplitGain cannot be set without setting linear to be true.")

    if OOBhonest and (splitratio != 1):
        warnings.warn("OOBhonest is set to true, so we will run OOBhonesty rather than standard honesty.")
        splitratio = 1

    if OOBhonest and not replace:
        warnings.warn("replace must be set to TRUE to use OOBhonesty, setting this to True now")
        replace = True

    if doubleTree and (splitratio == 0 or splitratio == 1):
        warnings.warn("Trees cannot be doubled if splitratio is 1. We have set doubleTree to False.")
        doubleTree = False

    if (interactionDepth is not None) and (maxDepth is not None) and (interactionDepth > maxDepth):
        warnings.warn("interactionDepth cannot be greater than maxDepth. We have set interactionDepth to maxDepth.")
        interactionDepth = maxDepth

    return splitratio, replace, doubleTree, interactionDepth


def training_data_checker(
    forest,
    x,
    y,
    linFeats,
    monotonicConstraints,
    groups,
    observationWeights,
    symmetric,
):
    """
    -- Sanity Checker -------------------------------------------------------------
    @name training_data_checker
    @title Training data check
    @rdname training_data_checker-RandomForest
    @description Check the input to RandomForest constructor
    @inheritParams RandomForest
    @param featureWeights weights used when subsampling features for nodes above or at interactionDepth.
    @param deepFeatureWeights weights used when subsampling features for nodes below interactionDepth.
    @param hasNas indicates if there is any missingness in x.
    @return A tuple of parameters after checking the selected parameters are valid.
    """

    nrows, nfeatures = x.shape

    # make np arrays
    symmetric = np.array(symmetric)

    # Check if the input dimension of x matches y
    if nrows != y.size:
        raise ValueError("The dimension of input dataset x doesn't match the output y.")

    if forest.linear and has_nas(x):
        raise ValueError("Cannot do imputation splitting with linear.")

    if np.isnan(y).any():
        raise ValueError("y contains missing data.")

    if any(i < 0 or i >= nfeatures for i in linFeats):
        raise ValueError("linFeats must contain positive integers less than len(x.columns).")

    if (not forest.replace) and forest.sampsize > nrows:
        raise ValueError("You cannot sample without replacement with size more than total number of observations.")

    if forest.mtry > nfeatures:
        raise ValueError("mtry cannot exceed total amount of features in x.")

    if monotonicConstraints.size != nfeatures:
        raise ValueError("monotonicConstraints must have the size of x")

    if any(i != 0 and i != 1 and i != -1 for i in monotonicConstraints):
        raise ValueError("monotonicConstraints must be either 1, 0, or -1")

    if any(i != 0 for i in monotonicConstraints) and forest.linear:
        raise ValueError("Cannot use linear splitting with monotonicConstraints")

    if observationWeights.size != nrows:
        raise ValueError("observationWeights must have length len(x)")

    if any(i < 0 for i in observationWeights):
        raise ValueError("The entries in observationWeights must be non negative")

    if forest.replace and np.sum(observationWeights) == 0:
        raise ValueError("There must be at least one non-zero weight in observationWeights")

    if any(i != 0 for i in symmetric):
        if forest.linear:
            raise ValueError(
                "Symmetric forests cannot be combined with linear aggregation please set either symmetric = False or linear = False"
            )

        if has_nas(x):
            raise ValueError(
                "Symmetric forests cannot be combined with missing values please impute the missing features before training a forest with symmetry"
            )

        if forest.scale:
            warnings.warn(
                "As symmetry is implementing pseudo outcomes, this causes problems when the Y values are scaled. Setting scale = False"
            )

        # for now don't scale when we run symmetric splitting since we use pseudo outcomes
        # and want to retain the scaling of Y
        forest.scale = False

        # OPTIMIZE ???
        if any(j != 1 and j != 0 for j in symmetric):
            raise ValueError("Entries of the symmetric argument must be zero one")

        if sum(j > 0 for j in symmetric) > 10:
            warnings.warn("Running symmetric splits in more than 10 features is very slow")

    # if the splitratio is 1, then we use adaptive rf and avgSampleSize is
    # equal to the total sampsize

    if forest.splitratio == 0 or forest.splitratio == 1:
        splitSampleSize = forest.sampsize
        avgSampleSize = forest.sampsize
    else:
        splitSampleSize = forest.splitratio * forest.sampsize
        avgSampleSize = math.floor(forest.sampsize - splitSampleSize)
        splitSampleSize = math.floor(splitSampleSize)

    if forest.nodesize_strict_spl > splitSampleSize:
        warnings.warn(
            "nodesizeStrictSpl cannot exceed splitting sample size. ",
            "We have set nodesizeStrictSpl to be the maximum.",
        )
        forest.nodesize_strict_spl = splitSampleSize

    if forest.nodesize_strict_avg > avgSampleSize:
        warnings.warn(
            "nodesizeStrictAvg cannot exceed averaging sample size. ",
            "We have set nodesizeStrictAvg to be the maximum.",
        )
        forest.nodesize_strict_avg = avgSampleSize

    if forest.double_tree:
        if forest.nodesize_strict_avg > splitSampleSize:
            warnings.warn(
                "nodesizeStrictAvg cannot exceed splitting sample size. ",
                "We have set nodesizeStrictAvg to be the maximum.",
            )
            forest.nodesize_strict_avg = splitSampleSize
        if forest.nodesize_strict_spl > avgSampleSize:
            warnings.warn(
                "nodesizeStrictSpl cannot exceed averaging sample size. ",
                "We have set nodesizeStrictSpl to be the maximum.",
            )
            forest.nodesize_strict_spl = avgSampleSize

    if groups is not None:
        if not pd.api.types.is_categorical_dtype(groups):
            raise ValueError(
                'groups must have a data dtype of categorical. Try using pd.Categorical(...) or pd.Series(..., dtype="category").'
            )
        if len(groups.unique()) == 1:
            raise ValueError("groups must have more than 1 level to be left out from sampling.")


def testing_data_checker(forest, newdata, has_nas):
    """
    @title Test data check
    @name testing_data_checker-RandomForest
    @description Check the testing data to do prediction
    @param forest A RandomForest object.
    @param newdata A data frame of testing predictors.
    @param hasNas TRUE if the there were nan-s in the training data FALSE otherwise.
    @return A feature dataframe if it can be used for new predictions.
    """

    if len(newdata.columns) != forest.processed_dta.num_columns:
        raise ValueError(
            "newdata has "
            + str(len(newdata.columns))
            + " but the forest was trained with "
            + str(forest.processed_dta.num_columns)
            + " columns."
        )

    if forest.processed_dta.feat_names is not None:
        if not set(newdata.columns) == set(forest.processed_dta.feat_names):
            raise ValueError("newdata has different columns then the ones the forest was trained with.")

        # If linear is true we can't predict observations with some features missing.
        if forest.linear and newdata.isnull().values.any():
            raise ValueError("linear does not support missing data")


def sample_weights_checker(feature_weights, mtry, ncol):
    if feature_weights.size != ncol:
        raise ValueError("featureWeights and deepFeatureWeights must have length len(x.columns)")

    if any(i < 0 for i in feature_weights):
        raise ValueError("The entries in featureWeights and deepFeatureWeights must be non negative")

    if np.sum(feature_weights) == 0:
        raise ValueError("There must be at least one non-zero weight in featureWeights and deepFeatureWeights")

    feature_weights_variables = [
        i for i in range(feature_weights.size) if feature_weights[i] > max(feature_weights) * 0.001
    ]
    if len(feature_weights_variables) < mtry:
        raise ValueError("mtry is too large. Given the feature weights, can't select that many features.")

    feature_weights_variables = np.array(feature_weights_variables, dtype=np.ulonglong)
    return feature_weights_variables


def forest_checker(forest):
    """
    Checks if RandomForest object has valid pointer for C++ object.
    @param object a RandomForest object
    @return A message if the forest does not have a valid C++ pointer.
    """

    if (not forest.dataframe) or (not forest.forest):
        raise ValueError("The RandomForest object has invalid ctypes pointers.")


def preprocess_training(x: pd.DataFrame, y) -> Tuple[pd.DataFrame, np.ndarray, List[Dict]]:
    """
    -- Methods for Preprocessing Data --------------------------------------------
    @title preprocess_training
    @description Perform preprocessing for the training data, including
      converting data to dataframe, and encoding categorical data into numerical
      representation.
    @inheritParams RandomForest
    @return A list of two datasets along with necessary information that encodes
      the preprocessing.
    """

    # Check if the input dimension of x matches y
    if len(x.index) != y.size:
        raise ValueError("The dimension of input dataset x doesn't match the output vector y.")

    # Track the order of all features
    feature_names = x.columns.values
    if feature_names.size == 0:
        warnings.warn("No names are given for each column.")

    # Track all categorical features (both factors and characters)
    categorical_feature_cols = np.array((x.select_dtypes("category")).columns)
    feature_character_cols = np.array((x.select_dtypes("object")).columns)

    if feature_character_cols.size != 0:  # convert to a factor
        warnings.warn("Character value features will be cast to categorical data.")
        categorical_feature_cols = np.concatenate((categorical_feature_cols, feature_character_cols), axis=0)

    categorical_feature_cols = x.columns.get_indexer(categorical_feature_cols)

    # For each categorical feature, encode x into numeric representation and
    # save the encoding mapping
    categorical_feature_mapping: List[Dict] = []
    for categorical_feature_col in categorical_feature_cols:
        x.iloc[:, categorical_feature_col] = pd.Series(
            x.iloc[:, categorical_feature_col], dtype="category"
        ).cat.remove_unused_categories()

        categorical_feature_mapping.append(
            {
                "categoricalFeatureCol": categorical_feature_col,
                "uniqueFeatureValues": list(x.iloc[:, categorical_feature_col].cat.categories),
                "numericFeatureValues": np.arange(len(x.iloc[:, categorical_feature_col].cat.categories)),
            }
        )

        x.iloc[:, categorical_feature_col] = pd.Series(x.iloc[:, categorical_feature_col].cat.codes, dtype="category")

    return x, categorical_feature_cols, categorical_feature_mapping


def preprocess_testing(x, categorical_feature_cols: np.ndarray, categorical_feature_mapping: List[Dict]) -> Any:
    """
    @title preprocess_testing
    @description Perform preprocessing for the testing data, including converting
      data to dataframe, and testing if the columns are consistent with the
      training data and encoding categorical data into numerical representation
      in the same way as training data.
    @inheritParams RandomForest
    @param categorical_feature_cols A list of index for all categorical data. Used
      for trees to detect categorical columns.
    @param categorical_feature_mapping A list of encoding details for each
      categorical column, including all unique factor values and their
      corresponding numeric representation.
    @return A preprocessed training dataaset x
    """

    # Track the order of all features
    testing_feature_names = x.columns.values
    if testing_feature_names.size == 0:
        warnings.warn("No names are given for each column.")

    # Track all categorical features (both factors and characters)
    feature_factor_cols = np.array((x.select_dtypes("category")).columns)
    feature_character_cols = np.array((x.select_dtypes("object")).columns)

    testing_categorical_feature_cols = np.concatenate((feature_factor_cols, feature_character_cols), axis=0)
    testing_categorical_feature_cols = x.columns.get_indexer(testing_categorical_feature_cols)

    if (set(categorical_feature_cols) - set(testing_categorical_feature_cols)) or (
        set(testing_categorical_feature_cols) - set(categorical_feature_cols)
    ):
        raise ValueError("Categorical columns are different between testing and training data.")

    # For each categorical feature, encode x into numeric representation
    for categorical_feature_mapping_ in categorical_feature_mapping:
        categorical_feature_col = categorical_feature_mapping_["categoricalFeatureCol"]
        # Get all unique feature values
        testing_unique_feature_values = x.iloc[:, categorical_feature_col].unique()
        unique_feature_values = categorical_feature_mapping_["uniqueFeatureValues"]
        numeric_feature_values = categorical_feature_mapping_["numericFeatureValues"]

        # If testing dataset contains more, adding new factors to the mapping list
        diff_unique_feature_values = set(testing_unique_feature_values) - set(unique_feature_values)
        if diff_unique_feature_values:
            unique_feature_values = np.concatenate(
                (list(unique_feature_values), list(diff_unique_feature_values)), axis=0
            )
            numeric_feature_values = np.arange(unique_feature_values.size)

            # update
            categorical_feature_mapping_["uniqueFeatureValues"] = unique_feature_values
            categorical_feature_mapping_["numericFeatureValues"] = numeric_feature_values

        x.iloc[:, categorical_feature_col] = pd.Series(
            find_match(x.iloc[:, categorical_feature_col], unique_feature_values),
            dtype="category",
        )

    # Return transformed data and encoding information
    return x


def scale_center(
    x: pd.DataFrame, categorical_feature_cols: np.ndarray, col_means: np.ndarray, col_sd: np.ndarray
) -> pd.DataFrame:
    """
    @title scale_center
    @description Given a dataframe, scale and center the continous features
    @param x A dataframe in order to be processed.
    @param categoricalFeatureCols A vector of the categorical features, we
      don't want to scale/center these.
    @param colMeans A vector of the means to center each column.
    @param colSd A vector of the standard deviations to scale each column with.
    @return A scaled and centered  dataset x
    """

    for col_idx in range(len(x.columns)):
        if col_idx not in categorical_feature_cols:
            if col_sd[col_idx] != 0:
                x.iloc[:, col_idx] = (x.iloc[:, col_idx] - col_means[col_idx]) / col_sd[col_idx]
            else:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] - col_means[col_idx]

    return x


def unscale_uncenter(x: pd.DataFrame, categorical_feature_cols: list, col_means: list, col_sd: list) -> pd.DataFrame:
    """
    @title unscale_uncenter
    @description Given a dataframe, un scale and un center the continous features
    @param x A dataframe in order to be processed.
    @param categoricalFeatureCols A vector of the categorical features, we
      don't want to scale/center these. Should be 1-indexed.
    @param colMeans A vector of the means to add to each column.
    @param colSd A vector of the standard deviations to rescale each column with.
    @return A dataset x in it's original scaling
    """

    for col_idx in range(len(x.columns)):
        if x.columns[col_idx] not in categorical_feature_cols:
            if col_sd[col_idx] != 0:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] * col_sd[col_idx] + col_means[col_idx]
            else:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] + col_means[col_idx]

    return x


def scale(x, y, processed_x, categorical_feature_cols):
    _, ncol = x.shape
    col_means = np.repeat(0.0, ncol + 1)
    col_sd = np.repeat(0.0, ncol + 1)

    for col_idx in range(ncol):
        if col_idx not in categorical_feature_cols:
            col_means[col_idx] = np.nanmean(processed_x.iloc[:, col_idx])
            col_sd[col_idx] = np.nanstd(processed_x.iloc[:, col_idx])

    # Scale columns of X
    processed_x = scale_center(processed_x, categorical_feature_cols, col_means, col_sd)

    # Center and scale Y
    col_means[ncol] = np.nanmean(y)
    col_sd[ncol] = np.nanstd(y)
    if col_sd[ncol] != 0:
        y = (y - col_means[ncol]) / col_sd[ncol]
    else:
        y = y - col_means[ncol]

    return processed_x, y, col_means, col_sd
