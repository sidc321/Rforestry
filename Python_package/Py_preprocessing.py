import pandas as pd
import numpy as np
import warnings



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
    featureNames = list(x.columns)
    if not featureNames:
        warnings.warn('No names are given for each column.')

    # Track all categorical features (both factors and characters)

    #################################
    return (x, [], [])
