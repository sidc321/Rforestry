import numpy as np
import pandas as pd
import warnings
import math
# --------------------------------------


#' @include R_preprocessing.R
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
		raise ValueError('The dimension of input dataset x doesn\'t match the output vector y.')

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
        raise ValueError('linFeats must contain positive integers less than ncol(x).')

    if (not replace) and sampsize > nrows:
        raise ValueError('You cannot sample without replacement with size more than total number of observations.')

    if (not isinstance(mtry, int)) or mtry <= 0:
		raise ValueError('mtry must be a positive integer.')

    if mtry > nfeatures:
        raise ValueError('mtry cannot exceed total amount of features in x.')

    if (not isinstance(nodesizeSpl, int)) or nodesizeSpl <= 0:
		raise ValueError('nodesizeSpl must be a positive integer.')
    
    if (not isinstance(nodesizeAvg, int)) or nodesizeAvg < 0:
		raise ValueError('nodesizeAvg must be a positive integer.')

    if (not isinstance(nodesizeStrictSpl, int)) or nodesizeStrictSpl <= 0:
		raise ValueError('nodesizeStrictSpl must be a positive integer.')

    if (not isinstance(nodesizeStrictAvg, int)) or nodesizeStrictAvg < 0:
		raise ValueError('nodesizeStrictAvg must be a positive integer.')

    if minSplitGain < 0:
        raise ValueError('minSplitGain must be greater than or equal to 0.')

    if minSplitGain > 0 and !linear: 
        raise ValueError('minSplitGain cannot be set without setting linear to be true.')

    if (not isinstance(maxDepth, int)) or maxDepth <= 0:
		raise ValueError('maxDepth must be a positive integer.')

    if (not isinstance(interactionDepth, int)) or interactionDepth <= 0:
		raise ValueError('interactionDepth must be a positive integer.')

    if len(monotonicConstraints) != nfeatures:
        raise ValueError('monotonicConstraints must be the size of x')

    if any(i != 0 and i != 1 and i != -1 for i in monotonicConstraints)::
        raise ValueError('monotonicConstraints must be either 1, 0, or -1')
  
    if any(i != 0 for i in monotonicConstraints) and linear:
        raise ValueError('Cannot use linear splitting with monotonicConstraints')

    if not replace:
        observationWeights = [1 for _ in range(nrows)]

    if len(observationWeights) != nrows:
        raise ValueError('observationWeights must have length nrow(x)')

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
        if any(j != 1 for j in symmetric):
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
        if splitratio == 0 || splitratio == 1:
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
        #FACTOR CONVERSION ISSUES

        if len(set(groups)) == 1:
            raise ValueError('groups must have more than 1 level to be left out from sampling.')

    if OOBhonest and (splitratio != 1):
        warnings.warn('OOBhonest is set to true, so we will run OOBhonesty rather than standard honesty.')
        splitratio = 1

    if OOBhonest and (replace == False):
        warnings.warn('replace must be set to TRUE to use OOBhonesty, setting this to True now')
        replace = True

    if (not isinstance(nthread, int)) or nthread < 0:
		raise ValueError('nthread must be a nonegative integer.')

    #????????????
    """
    if (nthread > 0) {
    #' @import parallel
    if (tryCatch(
      nthread > parallel::detectCores(),
      error = function(x) {
        FALSE
      }
    )) {
      stop(paste0(
        "nthread cannot exceed total cores in the computer: ",
        detectCores()
      ))
        }
    }
    """

    if not isinstance(middleSplit, bool):
        raise ValueError('middleSplit must be True or False.')


    return {
        'x': x,
        'y': y,
        'ntree': ntree,
        'replace': replace,
        'sampsize': sampsize,
        'mtry': mtry,
        'nodesizeSpl': nodesizeSpl,
        'nodesizeAvg': nodesizeAvg,
        'nodesizeStrictSpl': nodesizeStrictSpl,
        'nodesizeStrictAvg': nodesizeStrictAvg,
        'minSplitGain': minSplitGain,
        'maxDepth': maxDepth,
        'interactionDepth': interactionDepth,
        'splitratio': splitratio,
        'OOBhonest': OOBhonest,
        'nthread': nthread,
        'groups': groups,
        'middleSplit': middleSplit,
        'doubleTree': doubleTree,
        'linFeats': linFeats,
        'monotonicConstraints': monotonicConstraints,
        'featureWeights': featureWeights,
        "scale": scale,
        'deepFeatureWeights': deepFeatureWeights,
        'observationWeights': observationWeights,
        'hasNas': hasNas}


#' @title Test data check
#' @name testing_data_checker-forestry
#' @description Check the testing data to do prediction
#' @param object A forestry object.
#' @param newdata A data frame of testing predictors.
#' @param hasNas TRUE if the there were NAs in the training data FALSE otherwise.
#' @return A feature dataframe if it can be used for new predictions.
def testing_data_checker(object, newdata, hasNas):
	pass



def sample_weights_checker(featureWeights, mtry, ncol):
	if len(featureWeights) != ncol:
        raise ValueError('featureWeights and deepFeatureWeights must have length ncol(x)')

    if any(i < 0 for i in featureWeights):
        raise ValueError('The entries in featureWeights and deepFeatureWeights must be non negative')

    if sum(featureWeights) == 0:
        raise ValueError('There must be at least one non-zero weight in featureWeights and deepFeatureWeights')

    #AND THE REST


def forest_checker(object):
	pass





# -- Random Forest Constructor -------------------------------------------------
class forestry:

	pass


class multilayerForestry:

	pass