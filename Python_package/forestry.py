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

import Py_preprocessing

"""
--------------------------------------

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
        raise ValueError('replace must be True or False.')

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
            raise ValueError('groups must have a data dtype of categorical. Try using pd.Series(..., dtype="category") or pd.Categorical(...).')
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



# -- Random Forest Constructor -------------------------------------------------

"""
@title forestry
@rdname forestry
@param x A data frame of all training predictors.
@param y A list/array of all training responses.
@param ntree The number of trees to grow in the forest. The default value is
  500.
@param replace An indicator of whether sampling of training data is with
  replacement. The default value is TRUE.
@param sampsize The size of total samples to draw for the training data. If
  sampling with replacement, the default value is the length of the training
  data. If sampling without replacement, the default value is two-thirds of
  the length of the training data.
@param sample_fraction If this is given, then sampsize is ignored and set to
  be round(length(y) * sample_fraction). It must be a real number between 0 and 1
@param mtry The number of variables randomly selected at each split point.
  The default value is set to be one-third of the total number of features of the training data.
@param nodesizeSpl Minimum observations contained in terminal nodes.
  The default value is 5.
@param nodesizeAvg Minimum size of terminal nodes for averaging dataset.
  The default value is 5.
@param nodesizeStrictSpl Minimum observations to follow strictly in terminal nodes.
  The default value is 1.
@param nodesizeStrictAvg The minimum size of terminal nodes for averaging data set to follow when predicting.
  No splits are allowed that result in nodes with observations less than this parameter.
  This parameter enforces overlap of the averaging data set with the splitting set when training.
  When using honesty, splits that leave less than nodesizeStrictAvg averaging
  observations in either child node will be rejected, ensuring every leaf node
  also has at least nodesizeStrictAvg averaging observations. The default value is 1.
@param minSplitGain Minimum loss reduction to split a node further in a tree.
@param maxDepth Maximum depth of a tree. The default value is 99.
@param interactionDepth All splits at or above interaction depth must be on
  variables that are not weighting variables (as provided by the interactionVariables argument).
@param interactionVariables Indices of weighting variables.
@param featureWeights (optional) vector of sampling probabilities/weights for each
  feature used when subsampling mtry features at each node above or at interactionDepth.
  The default is to use uniform probabilities.
@param deepFeatureWeights Used in place of featureWeights for splits below interactionDepth.
@param observationWeights Denotes the weights for each training observation
  that determine how likely the observation is to be selected in each bootstrap sample.
  This option is not allowed when sampling is done without replacement.
@param splitratio Proportion of the training data used as the splitting dataset.
  It is a ratio between 0 and 1. If the ratio is 1 (the default), then the splitting
  set uses the entire data, as does the averaging set---i.e., the standard Breiman RF setup.
  If the ratio is 0, then the splitting data set is empty, and the entire dataset is used
  for the averaging set (This is not a good usage, however, since there will be no data available for splitting).
@param OOBhonest In this version of honesty, the out-of-bag observations for each tree
  are used as the honest (averaging) set. This setting also changes how predictions
  are constructed. When predicting for observations that are out-of-sample
  (using predict(..., aggregation = "average")), all the trees in the forest
  are used to construct predictions. When predicting for an observation that was in-sample (using
  predict(..., aggregation = "oob")), only the trees for which that observation
  was not in the averaging set are used to construct the prediction for that observation.
  aggregation="oob" (out-of-bag) ensures that the outcome value for an observation
  is never used to construct predictions for a given observation even when it is in sample.
  This property does not hold in standard honesty, which relies on an asymptotic
  subsampling argument. By default, when OOBhonest = TRUE, the out-of-bag observations
  for each tree are resamples with replacement to be used for the honest (averaging)
  set. This results in a third set of observations that are left out of both
  the splitting and averaging set, we call these the double out-of-bag (doubleOOB)
  observations. In order to get the predictions of only the trees in which each
  observation fell into this doubleOOB set, one can run predict(... , aggregation = "doubleOOB").
  In order to not do this second bootstrap sample, the doubleBootstrap flag can
  be set to FALSE.
@param doubleBootstrap The doubleBootstrap flag provides the option to resample
  with replacement from the out-of-bag observations set for each tree to construct
  the averaging set when using OOBhonest. If this is FALSE, the out-of-bag observations
  are used as the averaging set. By default this option is TRUE when running OOBhonest = TRUE.
  This option increases diversity across trees.
@param seed random seed
@param verbose Indicator to train the forest in verbose mode
@param nthread Number of threads to train and predict the forest. The default
  number is 0 which represents using all cores.
@param splitrule Only variance is implemented at this point and it
  specifies the loss function according to which the splits of random forest
  should be made.
@param middleSplit Indicator of whether the split value is takes the average of two feature
  values. If FALSE, it will take a point based on a uniform distribution
  between two feature values. (Default = FALSE)
@param doubleTree if the number of tree is doubled as averaging and splitting
  data can be exchanged to create decorrelated trees. (Default = FALSE)
@param reuseforestry Pass in an `forestry` object which will recycle the
  dataframe the old object created. It will save some space working on the
  same data set.
@param maxObs The max number of observations to split on.
@param savable If TRUE, then RF is created in such a way that it can be
  saved and loaded using save(...) and load(...). However, setting it to TRUE
  (default) will take longer and use more memory. When
  training many RF, it makes sense to set this to FALSE to save time and memory.
@param saveable deprecated. Do not use.
@param linear Indicator that enables Ridge penalized splits and linear aggregation
  functions in the leaf nodes. This is recommended for data with linear outcomes.
  For implementation details, see: https://arxiv.org/abs/1906.06463. Default is FALSE.
@param symmetric Used for the experimental feature which imposes strict symmetric
  marginal structure on the predictions of the forest through only selecting
  symmetric splits with symmetric aggregation functions. Should be a vector of size ncol(x) with a single
  1 entry denoting the feature to enforce symmetry on. Defaults to all zeroes.
  For version >= 0.9.0.83, we experimentally allow more than one feature to
  enforce symmetry at a time. This should only be used for a small number of
  features as it has a runtime that is exponential in the number of symmetric
  features (O(N 2^|S|) where S is the set of symmetric features).
@param linFeats A vector containing the indices of which features to split
  linearly on when using linear penalized splits (defaults to use all numerical features).
@param monotonicConstraints Specifies monotonic relationships between the continuous
  features and the outcome. Supplied as a vector of length p with entries in
  1,0,-1 which 1 indicating an increasing monotonic relationship, -1 indicating
  a decreasing monotonic relationship, and 0 indicating no constraint.
  Constraints supplied for categorical variable will be ignored.
@param groups A vector of factors specifying the group membership of each training observation.
  these groups are used in the aggregation when doing out of bag predictions in
  order to predict with only trees where the entire group was not used for aggregation.
  This allows the user to specify custom subgroups which will be used to create
  predictions which do not use any data from a common group to make predictions for
  any observation in the group. This can be used to create general custom
  resampling schemes, and provide predictions consistent with the Out-of-Group set.
@param minTreesPerGroup The number of trees which we make sure have been created leaving
  out each group. This is 0 by default, so we will not give any special treatment to
  the groups when sampling, however if this is set to a positive integer, we
  modify the bootstrap sampling scheme to ensure that exactly that many trees
  have the group left out. We do this by, for each group, creating minTreesPerGroup
  trees which are built on observations sampled from the set of training observations
  which are not in the current group. This means we create at least # groups * minTreesPerGroup
  trees for the forest. If ntree > # groups * minTreesPerGroup, we create
  max(# groups * minTreesPerGroup,ntree) total trees, in which at least minTreesPerGroup
  are created leaving out each group. For debugging purposes, these group sampling
  trees are stored at the end of the R forest, in blocks based on the left out group.
@param monotoneAvg This is a boolean flag that indicates whether or not monotonic
  constraints should be enforced on the averaging set in addition to the splitting set.
  This flag is meaningless unless both honesty and monotonic constraints are in use.
  The default is FALSE.
@param overfitPenalty Value to determine how much to penalize the magnitude
  of coefficients in ridge regression when using linear splits.
@param scale A parameter which indicates whether or not we want to scale and center
  the covariates and outcome before doing the regression. This can help with
  stability, so by default is TRUE.
@return A `forestry` object.
@examples
-------

In version 0.9.0.34, we have modified the handling of missing data. Instead of
the greedy approach used in previous iterations, we now test any potential
split by putting all NA's to the right, and all NA's to the left, and taking
the choice which gives the best MSE for the split. Under this version of handling
the potential splits, we will still respect monotonic constraints. So if we put all
NA's to either side, and the resulting leaf nodes have means which violate
the monotone constraints, the split will be rejected.
@export
"""
class forestry:

    def __init__(
        self,
        x,
        y,
        ntree = 500,
        replace = True,
        sampsize = None,  #Add a default value.
        sample_fraction = None,
        mtry = None,    #Add a default value.
        nodesizeSpl = 5,
        nodesizeAvg = 5,
        nodesizeStrictSpl = 1,
        nodesizeStrictAvg = 1,
        minSplitGain = 0,
        maxDepth = None,  #Add a default value.
        interactionDepth = None,   #Add a default value.
        interactionVariables = [],
        featureWeights = None,
        deepFeatureWeights = None,
        observationWeights = None,
        splitratio = 1,
        OOBhonest = False,
        doubleBootstrap = None, #Add a default value.
        seed = randrange(1001),
        verbose = False,
        nthread = 0,
        splitrule = 'variance',
        middleSplit = False,
        maxObs = None,    #Add a default value.
        linear = False,
        symmetric = None,   #Add a default value.
        linFeats = None,    #Add a default value.
        monotonicConstraints = None,    #Add a default value.
        groups = None,
        minTreesPerGroup = 0,
        monotoneAvg = False,
        overfitPenalty = 1,
        scale = True,
        doubleTree = False,
        reuseforestry = None,
        savable = True,
        saveable = True
    ):

        # Make sure that all the parameters exist when passed to forestry
        
        if isinstance(x, pd.DataFrame):
            featNames = x.columns.values

        elif type(x).__module__ == np.__name__ or isinstance(x, list) or isinstance(x, pd.Series):
            featNames = None
            print('x does not have column names. The check that columns are provided in the same order when training and predicting will be skipped', file=sys.stderr)

        else:
            raise AttributeError('x must be a Pandas DataFrame, a numpy array, or a regular list')

        x = pd.DataFrame(x)
        
        nrow, ncol = x.shape

        if sampsize is None:
            sampsize = nrow if replace else math.ceil(0.632 * nrow)

        # only if sample.fraction is given, update sampsize
        if sample_fraction is not None:
            sampsize = math.ceil(sample_fraction * nrow)

        # make linFeats unique
        if linFeats == None:
            linFeats = [i for i in range(ncol)]
        linFeats = list(set(linFeats))

        # Preprocess the data
        hasNas = x.isnull().values.any()

        # Create vectors with the column means and SD's for scaling
        colMeans = np.repeat(0, ncol+1)
        colSd = np.repeat(0, ncol+1)

        # Translating interactionVariables to featureWeights syntax
        if featureWeights is None:
            featureWeights = np.repeat(1, ncol)
            featureWeights[interactionVariables] = 0
        if deepFeatureWeights is None:
            deepFeatureWeights = np.repeat(1, ncol)
        if observationWeights is None:
            observationWeights = np.repeat(1, nrow)


        # Giving default values
        if mtry is None:
            mtry = max(int(ncol / 3), 1)

        if maxDepth is None:
            maxDepth = round(nrow / 2) + 1
        
        if interactionDepth is None:
            interactionDepth = maxDepth

        if doubleBootstrap is None:
            doubleBootstrap = OOBhonest

        if maxObs is None:
            maxObs = len(y)

        if symmetric is None:
            symmetric = np.repeat(0, ncol)

        if monotonicConstraints is None:
            monotonicConstraints = np.repeat(0, ncol)

        (x,
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
        hasNas) = training_data_checker(
            x = x,
            y = y,
            ntree = ntree,
            replace = replace,
            sampsize = sampsize,
            mtry = mtry,
            nodesizeSpl = nodesizeSpl,
            nodesizeAvg = nodesizeAvg,
            nodesizeStrictSpl = nodesizeStrictSpl,
            nodesizeStrictAvg = nodesizeStrictAvg,
            minSplitGain = minSplitGain,
            maxDepth = maxDepth,
            interactionDepth = interactionDepth,
            splitratio = splitratio,
            OOBhonest = OOBhonest,
            nthread = nthread,
            middleSplit = middleSplit,
            doubleTree = doubleTree,
            linFeats = linFeats,
            monotonicConstraints = monotonicConstraints,
            groups = groups,
            featureWeights = featureWeights,
            deepFeatureWeights = deepFeatureWeights,
            observationWeights = observationWeights,
            linear = linear,
            symmetric = symmetric,
            scale = scale,
            hasNas = hasNas
        )


        (featureWeightsVariables, featureWeights) = sample_weights_checker(featureWeights, mtry, ncol)
        (deepFeatureWeightsVariables, deepFeatureWeights) = sample_weights_checker(deepFeatureWeights, mtry, ncol)    
        

        groupsMapping = dict()
        if groups is not None:
            groupsMapping['groupValue'] = groups.cat.categories
            groupsMapping['groupNumericValue'] = np.arange(len(groups.cat.categories))

            groupVector = pd.to_numeric(groups)

            # Print warning if the group number and minTreesPerGroup results in a large forest
            if minTreesPerGroup > 0 and len(groups.cat.categories) * minTreesPerGroup > 2000:
                warnings.warn('Using ' + str(len(groups.cat.categories)) + ' groups with ' + str(minTreesPerGroup) + ' trees per group will train ' + str(len(groups.cat.categories) * minTreesPerGroup) + ' trees in the forest')
        
        else:
            groupVector = np.repeat(0, nrow)

        if reuseforestry is None:
            (processed_x, categoricalFeatureCols, categoricalFeatureMapping) =  Py_preprocessing.preprocess_training(x, y)
           
         
        
        else:
            pass


        



#SOME TESTING


df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                   columns=['a', 'b', 'c'])

y = [0, 0, 1]
res = training_data_checker(
    x = df2,
    y = y,
    ntree = 3,
    replace = True,
    sampsize = 4,
    mtry = 3,
    nodesizeSpl = 2,
    nodesizeAvg = 1,
    nodesizeStrictSpl = 2,
    nodesizeStrictAvg = 1,
    minSplitGain = 0,
    maxDepth = 10,
    interactionDepth = 8,
    splitratio = 0.5,
    OOBhonest = 0,
    nthread = 7,
    middleSplit = True,
    doubleTree = 0,
    linFeats = [0, 1, 2],
    monotonicConstraints = [0, 0, 0],
    groups = pd.Series([1,2,3], dtype="category"),
    featureWeights = 0,
    deepFeatureWeights = 0,
    observationWeights = [1, 2, 3],
    linear = False,
    symmetric = [0,1,0,1],
    scale = False,
    hasNas = False
)


PF1 = forestry(x=df2, y=y, groups=pd.Categorical([1,1,1,2,3,2,3,2,3,3,4,2,1,2,2,3,3,3,4]), minTreesPerGroup=100)