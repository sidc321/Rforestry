"""
LIBRARIES NEEDED
----------------------
"""

import numpy as np
import pandas as pd
import warnings
import math
import os
import sys
from random import randrange



"""
--------------------------------------

Helper Function
@return a nunpy array indicating the indices of first occurunces of 
  the elements of A in B
"""
def find_match(A, B):
    d = {}
    
    for i in range(len(B)):
        if not str(B[i]) in d:
            d[str(B[i])] = i
            
    return np.array([d[str(val)] for val in A])

"""
#' @include the preprocessign file
#-- Sanity Checker -------------------------------------------------------------
#' @name training_data_checker
#' @title Training data check
#' @rdname training_data_checker-forestry
#' @description Check the input to forestry constructor
#' @inheritParams forestry
#' @param featureWeights weights used when subsampling features for nodes above or at interactionDepth.
#' @param deepFeatureWeights weights used when subsampling features for nodes below interactionDepth.
#' @param hasNas indicates if there is any missingness in x.
#' @return A list of parameters after checking the selected parameters are valid.
"""
def training_data_checker(
    x,
    y,
    ntree,
    replace,
    sampsize,
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
    nthread,
    middleSplit,
    doubleTree,
    linFeats,
    monotonicConstraints,
    groups,
    featureWeights,
    deepFeatureWeights,
    observationWeights,
    linear,
    symmetric,
    scale,
    hasNas
):
    
    x = pd.DataFrame(x)
    y = np.array(y)
    nrows, nfeatures = x.shape

    # Check if the input dimension of x matches y
    if nrows != len(y):
        raise ValueError('The dimension of input dataset x doesn\'t match the output y.')

    if linear and hasNas:
        raise ValueError('Cannot do imputation splitting with linear.')

    if np.isnan(y).any():
        raise ValueError('y contains missing data.')

    if not isinstance(replace, bool):
        raise AttributeError('replace must be True or False.')

    if (not isinstance(ntree, int)) or ntree <= 0:
        raise ValueError('ntree must be a positive integer.')

    if (not isinstance(sampsize, int)) or sampsize <= 0:
        raise ValueError('sampsize must be a positive integer.')

    if any(i < 0 or i >= nfeatures for i in linFeats):
        raise ValueError('linFeats must contain positive integers less than len(x.columns).')

    if (not replace) and sampsize > nrows:
        raise ValueError('You cannot sample without replacement with size more than total number of observations.')

    if (not isinstance(mtry, int)) or mtry <= 0:
        raise ValueError('mtry must be a positive integer.')

    if mtry > nfeatures:
        raise ValueError('mtry cannot exceed total amount of features in x.')

    if (not isinstance(nodesizeSpl, int)) or nodesizeSpl <= 0:
        raise ValueError('nodesizeSpl must be a positive integer.')

    if (not isinstance(nodesizeAvg, int)) or nodesizeAvg <= 0:
        raise ValueError('nodesizeAvg must be a positive integer.')

    if (not isinstance(nodesizeStrictSpl, int)) or nodesizeStrictSpl <= 0:
        raise ValueError('nodesizeStrictSpl must be a positive integer.')

    if (not isinstance(nodesizeStrictAvg, int)) or nodesizeStrictAvg <= 0:
        raise ValueError('nodesizeStrictAvg must be a positive integer.')

    if minSplitGain < 0:
        raise ValueError('minSplitGain must be greater than or equal to 0.')

    if minSplitGain > 0 and (not linear):
        raise ValueError('minSplitGain cannot be set without setting linear to be true.')

    if (not isinstance(maxDepth, int)) or maxDepth <= 0:
        raise ValueError('maxDepth must be a positive integer.')

    if (not isinstance(interactionDepth, int)) or interactionDepth <= 0:
        raise ValueError('interactionDepth must be a positive integer.')

    if len(monotonicConstraints) != nfeatures:
        raise ValueError('monotonicConstraints must be the size of x')
    
    if any(i != 0 and i != 1 and i != -1 for i in monotonicConstraints):
        raise ValueError('monotonicConstraints must be either 1, 0, or -1')

    if any(i != 0 for i in monotonicConstraints) and linear:
        raise ValueError('Cannot use linear splitting with monotonicConstraints')

    if not replace:
        observationWeights = [1 for _ in range(nrows)]

    if len(observationWeights) != nrows:
        raise ValueError('observationWeights must have length len(x)')

    if any(i < 0 for i in observationWeights):
        raise ValueError('The entries in observationWeights must be non negative')

    if sum(observationWeights) == 0:
        raise ValueError('There must be at least one non-zero weight in observationWeights')

    if any(i != 0 for i in symmetric):
        if linear:
           raise ValueError('Symmetric forests cannot be combined with linear aggregation please set either symmetric = False or linear = False') 

        if hasNas:
            raise ValueError('Symmetric forests cannot be combined with missing values please impute the missing features before training a forest with symmetry')

        if scale:
            warnings.warn('As symmetry is implementing pseudo outcomes, this causes problems when the Y values are scaled. Setting scale = False')

        # for now don't scale when we run symmetric splitting since we use pseudo outcomes
        # and wnat to retain the scaling of Y
        scale = False

        #OPTIMIZE ???
        if any(j != 1 and j != 0 for j in symmetric):
            raise ValueError('Entries of the symmetric argument must be zero one')

        if sum(j > 0 for j in symmetric) > 10:
            warnings.warn('Running symmetric splits in more than 10 features is very slow')

    
    s = sum(observationWeights)
    observationWeights = [i/s for i in observationWeights]

    # if the splitratio is 1, then we use adaptive rf and avgSampleSize is
    # equal to the total sampsize

    if splitratio == 0 or splitratio == 1:
        splitSampleSize = sampsize
        avgSampleSize = sampsize
    else:
        splitSampleSize = splitratio * sampsize
        avgSampleSize = math.floor(sampsize - splitSampleSize)
        splitSampleSize = math.floor(splitSampleSize)

    if nodesizeStrictSpl > splitSampleSize:
        warnings.warn('nodesizeStrictSpl cannot exceed splitting sample size. We have set nodesizeStrictSpl to be the maximum.')
        nodesizeStrictSpl = splitSampleSize

    if nodesizeStrictAvg > avgSampleSize:
        warnings.warn('nodesizeStrictAvg cannot exceed averaging sample size. We have set nodesizeStrictAvg to be the maximum.')
        nodesizeStrictAvg = avgSampleSize


    if doubleTree:
        if splitratio == 0 or splitratio == 1:
            warnings.warn('Trees cannot be doubled if splitratio is 1. We have set doubleTree to False.')
            doubleTree = False
        else:
            if nodesizeStrictAvg > splitSampleSize:
                warnings.warn('nodesizeStrictAvg cannot exceed splitting sample size. We have set nodesizeStrictAvg to be the maximum.')
                nodesizeStrictAvg = splitSampleSize
            if nodesizeStrictSpl > avgSampleSize:
                warnings.warn('nodesizeStrictSpl cannot exceed averaging sample size. We have set nodesizeStrictSpl to be the maximum.')
                nodesizeStrictSpl = avgSampleSize

    if splitratio < 0 or splitratio > 1:
        raise ValueError('splitratio must in between 0 and 1.')

    if groups is not None:
        if not pd.api.types.is_categorical_dtype(groups):
            raise ValueError('groups must have a data dtype of categorical. Try using pd.Categorical(...) or pd.Series(..., dtype="category").')
        if len(groups.unique()) == 1:
            raise ValueError('groups must have more than 1 level to be left out from sampling.')
        groups = pd.Series(groups, dtype='category')

    if OOBhonest and (splitratio != 1):
        warnings.warn('OOBhonest is set to true, so we will run OOBhonesty rather than standard honesty.')
        splitratio = 1

    if OOBhonest and (replace == False):
        warnings.warn('replace must be set to TRUE to use OOBhonesty, setting this to True now')
        replace = True

    if nthread > 0:
        if nthread > os.cpu_count():
            raise ValueError('nthread cannot exceed total cores in the computer: ' + str(os.cpu_count()))
      
    if not isinstance(middleSplit, bool):
        raise ValueError('middleSplit must be True or False.')

    return (
        x,
        y,
        ntree,
        replace,
        sampsize,
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
        nthread,
        groups,
        middleSplit,
        doubleTree,
        linFeats,
        monotonicConstraints,
        featureWeights,
        scale,
        deepFeatureWeights,
        observationWeights,
        hasNas
    )


"""
@title Test data check
@name testing_data_checker-forestry
@description Check the testing data to do prediction
@param object A forestry object.
@param newdata A data frame of testing predictors.
@param hasNas TRUE if the there were nan-s in the training data FALSE otherwise.
@return A feature dataframe if it can be used for new predictions.
"""
def testing_data_checker(object, newdata, hasNas):
    pass



def sample_weights_checker(featureWeights, mtry, ncol):
    if len(featureWeights) != ncol:
        raise ValueError('featureWeights and deepFeatureWeights must have length len(x.columns)')

    if any(i < 0 for i in featureWeights):
        raise ValueError('The entries in featureWeights and deepFeatureWeights must be non negative')

    if sum(featureWeights) == 0:
        raise ValueError('There must be at least one non-zero weight in featureWeights and deepFeatureWeights')


    featureWeightsVariables = [i for i in range(len(featureWeights)) if featureWeights[i] > max(featureWeights)*0.001]
    if len(featureWeightsVariables) < mtry:
        featureWeights = []
    
    return (featureWeightsVariables, featureWeights)

"""
Checks if forestry object has valid pointer for C++ object.
@param object a forestry object
@return A message if the forest does not have a valid C++ pointer.
"""
def forest_checker(object):
    pass





# -- Methods for Preprocessing Data --------------------------------------------
#' @title preprocess_training
#' @description Perform preprocessing for the training data, including
#'   converting data to dataframe, and encoding categorical data into numerical
#'   representation.
#' @inheritParams forestry
#' @return A list of two datasets along with necessary information that encodes
#'   the preprocessing.

def preprocess_training(x,y):
    x = pd.DataFrame(x)
    # Check if the input dimension of x matches y
    if len(x.index) != len(y):
        raise ValueError('The dimension of input dataset x doesn\'t match the output vector y.')

    # Track the order of all features
    featureNames = np.array(x.columns)
    if featureNames.size == 0:
        warnings.warn('No names are given for each column.')

    # Track all categorical features (both factors and characters)

    categoricalFeatureCols = np.array((x.select_dtypes('category')).columns)
    featureCharacterCols = np.array((x.select_dtypes('object')).columns)
    # Note: this will select all the data types that are objects - pointers to another 
    # block, like strings, dictionaries, etc.

    if featureCharacterCols.size != 0:
        raise AttributeError('Character value features must be cast to factors.')

    # For each categorical feature, encode x into numeric representation and
    # save the encoding mapping
    categoricalFeatureMapping = [None for _ in range(categoricalFeatureCols.size)]
    dummyIndex = 0
    for categoricalFeatureCol in categoricalFeatureCols:
        x[categoricalFeatureCol] = pd.Series(x[categoricalFeatureCol], dtype='category').cat.remove_unused_categories()

        categoricalFeatureMapping[dummyIndex] = {
            'categoricalFeatureCol': categoricalFeatureCol,
            'uniqueFeatureValues' : list(x[categoricalFeatureCol].cat.categories),
            'numericFeatureValues': np.arange(len(x[categoricalFeatureCol].cat.categories))
        }

        x[categoricalFeatureCol] = pd.Series(x[categoricalFeatureCol].cat.codes, dtype='category')
        dummyIndex += 1

    
    return (x, categoricalFeatureCols, categoricalFeatureMapping)


#' @title preprocess_testing
#' @description Perform preprocessing for the testing data, including converting
#'   data to dataframe, and testing if the columns are consistent with the
#'   training data and encoding categorical data into numerical representation
#'   in the same way as training data.
#' @inheritParams forestry
#' @param categoricalFeatureCols A list of index for all categorical data. Used
#'   for trees to detect categorical columns.
#' @param categoricalFeatureMapping A list of encoding details for each
#'   categorical column, including all unique factor values and their
#'   corresponding numeric representation.
#' @return A preprocessed training dataaset x
def preprocess_testing (x, categoricalFeatureCols, categoricalFeatureMapping):
    x = pd.DataFrame(x)

    # Track the order of all features
    testingFeatureNames = np.array(x.columns)
    if testingFeatureNames.size == 0:
        warnings.warn('No names are given for each column.')
    
    # Track all categorical features (both factors and characters)
    featureFactorCols = np.array((x.select_dtypes('category')).columns)
    featureCharacterCols = np.array((x.select_dtypes('object')).columns)

    testingCategoricalFeatureCols = np.concatenate((featureFactorCols, featureCharacterCols), axis=0)
    
    if set(categoricalFeatureCols) - set(testingCategoricalFeatureCols):
        raise ValueError('Categorical columns are different between testing and training data.')

    # For each categorical feature, encode x into numeric representation
    for categoricalFeatureMapping_ in categoricalFeatureMapping:
        categoricalFeatureCol = categoricalFeatureMapping_['categoricalFeatureCol']
        # Get all unique feature values
        testingUniqueFeatureValues = x[categoricalFeatureCol].unique()
        uniqueFeatureValues = categoricalFeatureMapping_['uniqueFeatureValues']
        numericFeatureValues = categoricalFeatureMapping_['numericFeatureValues']

        # If testing dataset contains more, adding new factors to the mapping list
        diffUniqueFeatureValues = set(testingUniqueFeatureValues) - set(uniqueFeatureValues)
        if diffUniqueFeatureValues:
            uniqueFeatureValues = np.concatenate((list(uniqueFeatureValues), list(diffUniqueFeatureValues)), axis=0)
            numericFeatureValues = np.arange(uniqueFeatureValues.size)

            #update
            categoricalFeatureMapping_['uniqueFeatureValues'] = uniqueFeatureValues
            categoricalFeatureMapping_['numericFeatureValues'] = numericFeatureValues

        x[categoricalFeatureCol] = pd.Series(find_match(x[categoricalFeatureCol], uniqueFeatureValues), dtype='category')
    
    # Return transformed data and encoding information
    return x


#' @title scale_center
#' @description Given a dataframe, scale and center the continous features
#' @param x A dataframe in order to be processed.
#' @param categoricalFeatureCols A vector of the categorical features, we
#'   don't want to scale/center these. Should be 1-indexed.
#' @param colMeans A vector of the means to center each column.
#' @param colSd A vector of the standard deviations to scale each column with.
#' @return A scaled and centered  dataset x
def scale_center(x, categoricalFeatureCols, colMeans, colSd):
    for col_idx in range(len(x.columns)):
        if x.columns[col_idx] not in categoricalFeatureCols:
            if colSd[col_idx] != 0:
                x.iloc[:, col_idx] = (x.iloc[:, col_idx] - colMeans[col_idx]) / colSd[col_idx]
            else:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] - colMeans[col_idx]

    return x



#' @title unscale_uncenter
#' @description Given a dataframe, un scale and un center the continous features
#' @param x A dataframe in order to be processed.
#' @param categoricalFeatureCols A vector of the categorical features, we
#'   don't want to scale/center these. Should be 1-indexed.
#' @param colMeans A vector of the means to add to each column.
#' @param colSd A vector of the standard deviations to rescale each column with.
#' @return A dataset x in it's original scaling
def unscale_uncenter(x, categoricalFeatureCols, colMeans, colSd):
    for col_idx in range(len(x.columns)):
        if x.columns[col_idx] not in categoricalFeatureCols:
            if colSd[col_idx] != 0:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] * colSd[col_idx] + colMeans[col_idx]
            else:
                x.iloc[:, col_idx] = x.iloc[:, col_idx] + colMeans[col_idx]

    return x