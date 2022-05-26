"""
LIBRARIES NEEDED
----------------------
"""

from hashlib import new
import numpy as np
import pandas as pd
import warnings
import math
import os
import sys
from random import randrange
import ctypes
import lib_setup

import Py_preprocessing



# --- Loading the dynamic library -----------------

lib = (ctypes.CDLL("../src/libforestryCpp.so"))  #CHANGE TO DLL IF NECESSARY
#forestry = (ctypes.CDLL("src/libforestryCpp.dylib")) 
lib_setup.setup_lib(lib)

# -- Random Forest Constructor -------------------------------------------------

"""
@title forestry
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
    ##### NOTE: Remeve params x, y
    def __init__(
        self,
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
        minTreesPerGroup = 0,
        monotoneAvg = False,
        overfitPenalty = 1,
        scale = True,
        doubleTree = False,
        reuseforestry = None,
        savable = True,
        saveable = True
    ):


        if doubleBootstrap is None:
            doubleBootstrap = OOBhonest


        # Call the forest_parameter_checker function
        # haven't checking splitrule

        (ntree,
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
        doubleTree) = Py_preprocessing.forest_parameter_checker(
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
            doubleTree)

        if reuseforestry is None:
            
            cppDataFrame = None  #Cpp pointer
            cppForest = None  #Cpp pointer
            processed_dta = {
                'processed_x': None,
                'y': None,
                'categoricalFeatureCols_cpp': None,
                'categoricalFeatureMapping': None,
                'featureWeights': None,
                'featureWeightsVariables': None,
                'deepFeatureWeights': None,
                'deepFeatureWeightsVariables': None,
                'observationWeights': None,
                'symmetric': None,
                'monotonicConstraints': None,
                'linearFeatureCols_cpp': None,
                'groupsMapping': None,
                'groups': None,
                'colMeans': None,
                'colSd': None,
                'hasNas': None,
                'nObservations': None,
                'numColumns': None,
                'featNames': None
            }
            Py_forest = np.array([])  # for printing

            #Data Fields
            self.forest = cppForest
            self.dataframe = cppDataFrame
            self.processed_dta = processed_dta
            self.Py_forest = Py_forest
            self.ntree = ntree
            self.replace = replace
            self.sampsize = sampsize
            self.sample_fraction = sample_fraction
            self.mtry = mtry
            self.nodesizeSpl = nodesizeSpl
            self.nodesizeAvg = nodesizeAvg
            self.nodesizeStrictSpl = nodesizeStrictSpl
            self.nodesizeStrictAvg = nodesizeStrictAvg
            self.minSplitGain = minSplitGain
            self.maxDepth = maxDepth
            self.interactionDepth = interactionDepth
            self.splitratio = splitratio
            self.OOBhonest = OOBhonest
            self.doubleBootstrap = doubleBootstrap
            self.seed = seed
            self.verbose = verbose
            self.nthread = nthread
            self.middleSplit = middleSplit
            self.maxObs = maxObs
            self.linear = linear
            self.monotoneAvg = monotoneAvg
            self.overfitPenalty = overfitPenalty
            self.doubleTree = doubleTree
            self.scale = scale
            self.minTreesPerGroup = minTreesPerGroup

        # reuseforestry Forestry is not None
        else:
            pass
            #NEED TO FINISH THIS


    def fit(
        self,
        x,
        y,
        interactionVariables = [],
        featureWeights = None,
        deepFeatureWeights = None,
        observationWeights = None,
        symmetric = None,  #Add a default value.
        linFeats = None,   #Add a default value.
        monotonicConstraints = None,   #Add a default value.
        groups = None
    ):
        
         # Make sure that all the parameters exist when passed to forestry

        if isinstance(x, pd.DataFrame):
            featNames = x.columns.values

        elif type(x).__module__ == np.__name__ or isinstance(x, list) or isinstance(x, pd.Series):
            featNames = None
            print('x does not have column names. The check that columns are provided in the same order when training and predicting will be skipped', file=sys.stderr)

        else:
            raise AttributeError('x must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list')

        x = pd.DataFrame(x)
        y = np.array(y)
        
        nrow, ncol = x.shape
            
        if self.sampsize is None:
            self.sampsize = nrow if self.replace else math.ceil(0.632 * nrow)

        # only if sample.fraction is given, update sampsize
        if self.sample_fraction is not None:
            self.sampsize = math.ceil(self.sample_fraction * nrow)

        # make linFeats unique
        if linFeats == None:
            linFeats = np.arange(ncol)
        linFeats = pd.unique(linFeats)

        # Preprocess the data
        hasNas = x.isnull().values.any()

        # Create vectors with the column means and SD's for scaling
        colMeans = np.repeat(0.0, ncol+1)
        colSd = np.repeat(0.0, ncol+1)


        # Translating interactionVariables to featureWeights syntax
        if featureWeights is None:
            featureWeights = np.repeat(1, ncol)
            featureWeights[interactionVariables] = 0
        if deepFeatureWeights is None:
            deepFeatureWeights = np.repeat(1, ncol)
        if observationWeights is None:
            observationWeights = np.repeat(1, nrow)

        # Giving default values
        if self.mtry is None:
            self.mtry = max((ncol // 3), 1)

        if self.maxDepth is None:
            self.maxDepth = round(nrow / 2) + 1
        
        if self.interactionDepth is None:
            self.interactionDepth = self.maxDepth

        if self.maxObs is None:
            self.maxObs = y.size

        if symmetric is None:
            symmetric = np.repeat(0, ncol)

        if monotonicConstraints is None:
            monotonicConstraints = np.repeat(0, ncol)

        
        (x,
        y,
        linFeats,
        monotonicConstraints,
        groups,
        observationWeights,
        symmetric,
        hasNas) = Py_preprocessing.training_data_checker(
            self,
            x,
            y,
            linFeats,
            monotonicConstraints,
            groups,
            observationWeights,
            symmetric,
            hasNas)

        (featureWeightsVariables, featureWeights) = Py_preprocessing.sample_weights_checker(featureWeights, self.mtry, ncol)
        (deepFeatureWeightsVariables, deepFeatureWeights) = Py_preprocessing.sample_weights_checker(deepFeatureWeights, self.mtry, ncol)    
        
        groupsMapping = dict()
        if groups is not None:
            groupsMapping['groupValue'] = groups.cat.categories
            groupsMapping['groupNumericValue'] = np.arange(len(groups.cat.categories))

            groupVector = pd.to_numeric(groups)

            # Print warning if the group number and minTreesPerGroup results in a large forest
            if object.minTreesPerGroup > 0 and len(groups.cat.categories) * object.minTreesPerGroup > 2000:
                warnings.warn('Using ' + str(len(groups.cat.categories)) + ' groups with ' + str(object.minTreesPerGroup) + ' trees per group will train ' + str(len(groups.cat.categories) * object.minTreesPerGroup) + ' trees in the forest')
        
        else:
            groupVector = np.repeat(0, nrow)


        #IGNORING reuseforestry FOR NOW

        (processed_x, categoricalFeatureCols_cpp, categoricalFeatureMapping) =  Py_preprocessing.preprocess_training(x, y)
        
        if categoricalFeatureCols_cpp.size != 0:
            monotonicConstraints[categoricalFeatureCols_cpp] = 0
        

        if self.scale:
            for col_idx in range(ncol):
                if col_idx not in categoricalFeatureCols_cpp:
                    colMeans[col_idx] = np.nanmean(processed_x.iloc[:, col_idx])
                    colSd[col_idx] = np.nanstd(processed_x.iloc[:, col_idx])

            # Scale columns of X
            processed_x = Py_preprocessing.scale_center(processed_x, categoricalFeatureCols_cpp, colMeans, colSd)
                
            # Center and scale Y
            colMeans[ncol] = np.nanmean(y)
            colSd[ncol] = np.nanstd(y)
            if colSd[ncol] != 0:
                y = (y - colMeans[ncol]) / colSd[ncol]    
            else:
                y = y - colMeans[ncol]


        # Get the symmetric feature if one is set
        symmetricIndex = -1
        idxs = np.where(symmetric > 0)[0]
        if idxs.size != 0:
            symmetricIndex = idxs[0]  

        isSymmetric = symmetricIndex == -1

        # cpp linking
        X = pd.concat([processed_x, pd.Series(y)], axis=1)
        self.dataframe = ctypes.c_void_p(lib.get_data(
            lib_setup.get_data_pointer(X),
            lib_setup.get_array_pointer(categoricalFeatureCols_cpp, dtype=np.ulonglong), categoricalFeatureCols_cpp.size,
            lib_setup.get_array_pointer(linFeats, dtype=np.ulonglong), linFeats.size,
            lib_setup.get_array_pointer(featureWeights, dtype=np.double),
            lib_setup.get_array_pointer(featureWeightsVariables, dtype=np.ulonglong), featureWeightsVariables.size,
            lib_setup.get_array_pointer(observationWeights, dtype=np.double),
            lib_setup.get_array_pointer(monotonicConstraints, dtype=np.intc),
            lib_setup.get_array_pointer(groupVector, dtype=np.ulonglong),
            self.monotoneAvg,
            lib_setup.get_array_pointer(symmetric, dtype=np.ulonglong), symmetric.size,
            nrow, ncol+1,
            self.seed
        ))

        self.forest = ctypes.c_void_p(lib.train_forest(
            self.dataframe,
            self.ntree,
            self.replace,
            self.sampsize,
            self.splitratio,
            self.OOBhonest,
            self.doubleBootstrap,
            self.mtry,
            self.nodesizeSpl,
            self.nodesizeAvg,
            self.nodesizeStrictSpl,
            self.nodesizeStrictAvg,
            self.minSplitGain,
            self.maxDepth,
            self.interactionDepth,
            self.seed,
            self.nthread,
            self.verbose,
            self.middleSplit,
            self.maxObs,
            self.minTreesPerGroup,
            hasNas,
            self.linear,
            isSymmetric,
            self.overfitPenalty,
            self.doubleTree
        ))
        
        # Update the fields 

        self.processed_dta['processed_x'] = processed_x
        self.processed_dta['y'] = y
        self.processed_dta['categoricalFeatureCols_cpp'] = categoricalFeatureCols_cpp
        self.processed_dta['categoricalFeatureMapping'] = categoricalFeatureMapping
        self.processed_dta['featureWeights'] = featureWeights
        self.processed_dta['featureWeightsVariables'] = featureWeightsVariables
        self.processed_dta['deepFeatureWeights'] = deepFeatureWeights
        self.processed_dta['deepFeatureWeightsVariables'] = deepFeatureWeightsVariables
        self.processed_dta['observationWeights'] = observationWeights
        self.processed_dta['symmetric'] = symmetric
        self.processed_dta['monotonicConstraints'] = monotonicConstraints
        self.processed_dta['linearFeatureCols_cpp'] = linFeats
        self.processed_dta['groupsMapping'] = groupsMapping
        self.processed_dta['groups'] = groups
        self.processed_dta['colMeans'] = colMeans
        self.processed_dta['colSd'] = colSd
        self.processed_dta['hasNas'] = hasNas
        self.processed_dta['nObservations'] = nrow
        self.processed_dta['numColumns'] = ncol
        self.processed_dta['featNames'] = featNames



    # -- Predict Method ------------------------------------------------------------
    #' predict-forestry
    #' @name predict-forestry
    #' @rdname predict-forestry
    #' @description Return the prediction from the forest.
    #' @param object A `forestry` object.
    #' @param newdata A data frame of testing predictors.
    #' @param aggregation How the individual tree predictions are aggregated:
    #'   `average` returns the mean of all trees in the forest; `terminalNodes` also returns
    #'   the weightMatrix, as well as "terminalNodes", a matrix where
    #'   the ith entry of the jth column is the index of the leaf node to which the
    #'   ith observation is assigned in the jth tree; and "sparse", a matrix
    #'   where the ith entry in the jth column is 1 if the ith observation in
    #'   newdata is assigned to the jth leaf and 0 otherwise. In each tree the
    #'   leaves are indexed using a depth first ordering, and, in the "sparse"
    #'   representation, the first leaf in the second tree has column index one more than
    #'   the number of leaves in the first tree and so on. So, for example, if the
    #'   first tree has 5 leaves, the sixth column of the "sparse" matrix corresponds
    #'   to the first leaf in the second tree.
    #'   `oob` returns the out-of-bag predictions for the forest. We assume
    #'   that the ordering of the observations in newdata have not changed from
    #'   training. If the ordering has changed, we will get the wrong OOB indices.
    #'   `doubleOOB` is an experimental flag, which can only be used when OOBhonest = TRUE
    #'   and doubleBootstrap = TRUE. When both of these settings are on, the
    #'   splitting set is selected as a bootstrap sample of observations and the
    #'   averaging set is selected as a bootstrap sample of the observations which
    #'   were left out of bag during the splitting set selection. This leaves a third
    #'   set which is the observations which were not selected in either bootstrap sample.
    #'   This predict flag gives the predictions using- for each observation- only the trees
    #'   in which the observation fell into this third set (so was neither a splitting
    #'   nor averaging example).
    #'   `coefs` is an aggregation option which works only when linear aggregation
    #'   functions have been used. This returns the linear coefficients for each
    #'   linear feature which were used in the leaf node regression of each predicted
    #'   point.
    #' @param seed random seed
    #' @param nthread The number of threads with which to run the predictions with.
    #'   This will default to the number of threads with which the forest was trained
    #'   with.
    #' @param exact This specifies whether the forest predictions should be aggregated
    #'   in a reproducible ordering. Due to the non-associativity of floating point
    #'   addition, when we predict in parallel, predictions will be aggregated in
    #'   varied orders as different threads finish at different times.
    #'   By default, exact is TRUE unless N > 100,000 or a custom aggregation
    #'   function is used.
    #' @param trees A vector of indices in the range 1:ntree which tells
    #'   predict which trees in the forest to use for the prediction. Predict will by
    #'   default take the average of all trees in the forest, although this flag
    #'   can be used to get single tree predictions, or averages of diffferent trees
    #'   with different weightings. Duplicate entries are allowed, so if trees = c(1,2,2)
    #'   this will predict the weighted average prediction of only trees 1 and 2 weighted by:
    #'   predict(..., trees = c(1,2,2)) = (predict(..., trees = c(1)) +
    #'                                      2*predict(..., trees = c(2))) / 3.
    #'   note we must have exact = TRUE, and aggregation = "average" to use tree indices.
    #' @param weightMatrix An indicator of whether or not we should also return a
    #'   matrix of the weights given to each training observation when making each
    #'   prediction. When getting the weight matrix, aggregation must be one of
    #'   `average`, `oob`, and `doubleOOB`.
    #' @param ... additional arguments.
    #' @return A vector of predicted responses.
    #' @export  

    def predict(self, newdata=None, aggregation = 'average', seed = None, nthread = 8, exact = None, trees = None, weightMatrix = False):
        if (newdata is None) and not (aggregation == 'oob' or aggregation == 'doubleOOB'):
            raise ValueError('When using an aggregation that is not oob or doubleOOB, one must supply newdata')

        if (not self.linear) and aggregation == 'coefs':
            raise ValueError('Aggregation can only be linear with setting the parameter linear = TRUE.')


        # Preprocess the data. We only run the data checker if ridge is turned on,
        # because even in the case where there were no NAs in train, we still want to predict.

        if newdata is not None:
            Py_preprocessing.forest_checker(self)

            if not (isinstance(newdata, pd.DataFrame) or type(newdata).__module__ == np.__name__ or isinstance(newdata, list) or isinstance(newdata, pd.Series)):
                raise AttributeError('newdata must be a Pandas DataFrame, a numpy array, a Pandas Series, or a regular list')

            newdata = pd.DataFrame(newdata)
            newdata = Py_preprocessing.testing_data_checker(self, newdata, self.processed_dta['hasNas'])
            
            processed_x = Py_preprocessing.preprocess_testing(newdata, self.processed_dta['categoricalFeatureCols_cpp'], self.processed_dta['categoricalFeatureMapping'])

        # Set exact aggregation method if nobs < 100,000 and average aggregation
        if exact is None:
            if (newdata is not None) and len(newdata.index) > 1e5:
                exact = False
            else:
                exact = True
    
        # We can only use tree aggregations if exact = TRUE and aggregation = "average"
        if (trees is not None) and ((not exact) or (aggregation != 'average')):
            raise ValueError('When using tree indices, we must have exact = True and aggregation = \'average\' ')

        if any((not isinstance(i, int)) or (i < 0) or (i >= self.ntree) for i in trees):
            raise ValueError('trees must contain indices which are integers between 1 and ntree')

        # If trees are being used, we need to convert them into a weight vector
        tree_weights = np.repeat(0, self.ntree)
        if trees is not None:
            for i in range(len(trees)):
                tree_weights[trees[i]] += 1
            use_weights = True
        else:
            use_weights = False


        # If option set to terminalNodes, we need to make matrix of ID's
        if aggregation == 'oob':

            if (newdata is not None) and (self.processed_dta['nObservations'] != len(newdata.index)):
                warnings.warn('Attempting to do OOB predictions on a dataset which doesn\'t match the training data!')
                return None

            if newdata is None:
                #Cpp oob prediction
                pass

            else:
                #Cpp
                pass
        

        forest_preds = ctypes.c_void_p(lib.predict_forest(
            self.forest,
            self.dataframe,
            lib_setup.get_data_pointer(processed_x),
            len(processed_x.index),
            self.verbose
        ))

        res = np.empty(len(processed_x.index))
        for i in range(len(processed_x.index)):
            res[i] = lib.vector_get(forest_preds, i)
        
        return res




#make linFeats same as symmetric...