// [[Rcpp::plugins(cpp11)]]
#include "DataFrame.h"
#include "forestryTree.h"
#include "RFNode.h"
#include "forestry.h"
#include "multilayerForestry.h"
#include "utils.h"
#include <RcppArmadillo.h>

void freeforestry(
  SEXP ptr
){
  if (NULL == R_ExternalPtrAddr(ptr))
    return;
  forestry* pm = (forestry*)(R_ExternalPtrAddr(ptr));
  delete(pm);
  R_ClearExternalPtr(ptr);
}

void freeMultilayerForestry(
    SEXP ptr
){
  if (NULL == R_ExternalPtrAddr(ptr))
    return;
  multilayerForestry* pm = (multilayerForestry*)(R_ExternalPtrAddr(ptr));
  delete(pm);
  R_ClearExternalPtr(ptr);
}

// [[Rcpp::export]]
SEXP rcpp_cppDataFrameInterface(
    Rcpp::List x,
    Rcpp::NumericVector y,
    Rcpp::NumericVector catCols,
    Rcpp::NumericVector linCols,
    int numRows,
    int numColumns,
    Rcpp::NumericVector featureWeights,
    Rcpp::NumericVector featureWeightsVariables,
    Rcpp::NumericVector deepFeatureWeights,
    Rcpp::NumericVector deepFeatureWeightsVariables,
    Rcpp::NumericVector observationWeights,
    Rcpp::NumericVector monotonicConstraints,
    Rcpp::NumericVector groupMemberships,
    bool monotoneAvg,
    Rcpp::NumericVector symmetricIndices
){

  try {
    std::unique_ptr<std::vector< std::vector<double> > > featureDataRcpp (
        new std::vector< std::vector<double> >(
            Rcpp::as< std::vector< std::vector<double> > >(x)
        )
    );

    std::unique_ptr< std::vector<double> > outcomeDataRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(y)
        )
    );

    std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
        new std::vector< size_t>(
            Rcpp::as< std::vector<size_t> >(catCols)
        )
    );

    std::unique_ptr< std::vector<size_t> > linearFeats (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(linCols)
        )
    );

    std::sort(linearFeats->begin(), linearFeats->end());

    std::unique_ptr< std::vector<double> > featureWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(featureWeights)
        )
    );

    std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
        )
    );

    std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(deepFeatureWeights)
        )
    );

    std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
        )
    );

    std::unique_ptr< std::vector<double> > observationWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(observationWeights)
        )
    );

    std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp (
        new std::vector<int>(
            Rcpp::as< std::vector<int> >(monotonicConstraints)
        )
    );

    std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(groupMemberships)
        )
    );

    std::unique_ptr< std::vector<size_t> > symmetricIndicesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(symmetricIndices)
        )
    );

    DataFrame* trainingData = new DataFrame(
        std::move(featureDataRcpp),
        std::move(outcomeDataRcpp),
        std::move(categoricalFeatureColsRcpp),
        std::move(linearFeats),
        (size_t) numRows,
        (size_t) numColumns,
        std::move(featureWeightsRcpp),
        std::move(featureWeightsVariablesRcpp),
        std::move(deepFeatureWeightsRcpp),
        std::move(deepFeatureWeightsVariablesRcpp),
        std::move(observationWeightsRcpp),
        std::move(monotonicConstraintsRcpp),
        std::move(groupMembershipsRcpp),
        (bool) monotoneAvg,
        std::move(symmetricIndicesRcpp)
    );

    Rcpp::XPtr<DataFrame> ptr(trainingData, true) ;
    return ptr;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return NULL;
}


// [[Rcpp::export]]
SEXP rcpp_cppBuildInterface(
  Rcpp::List x,
  Rcpp::NumericVector y,
  Rcpp::NumericVector catCols,
  Rcpp::NumericVector linCols,
  int numRows,
  int numColumns,
  int ntree,
  bool replace,
  int sampsize,
  int mtry,
  double splitratio,
  bool OOBhonest,
  bool doubleBootstrap,
  int nodesizeSpl,
  int nodesizeAvg,
  int nodesizeStrictSpl,
  int nodesizeStrictAvg,
  double minSplitGain,
  int maxDepth,
  int interactionDepth,
  int seed,
  int nthread,
  bool verbose,
  bool middleSplit,
  int maxObs,
  Rcpp::NumericVector featureWeights,
  Rcpp::NumericVector featureWeightsVariables,
  Rcpp::NumericVector deepFeatureWeights,
  Rcpp::NumericVector deepFeatureWeightsVariables,
  Rcpp::NumericVector observationWeights,
  Rcpp::NumericVector monotonicConstraints,
  Rcpp::NumericVector groupMemberships,
  Rcpp::NumericVector symmetricIndices,
  int minTreesPerGroup,
  bool monotoneAvg,
  bool hasNas,
  bool linear,
  bool symmetric,
  double overfitPenalty,
  bool doubleTree,
  bool existing_dataframe_flag,
  SEXP existing_dataframe
){

  if (existing_dataframe_flag) {

    try {
      Rcpp::XPtr< DataFrame > trainingData(existing_dataframe) ;

      forestry* testFullForest = new forestry(
        trainingData,
        (size_t) ntree,
        replace,
        (size_t) sampsize,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (double) minSplitGain,
        (size_t) maxDepth,
        (size_t) interactionDepth,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        (size_t) minTreesPerGroup,
        hasNas,
        linear,
        symmetric,
        (double) overfitPenalty,
        doubleTree
      );

      // delete(testFullForest);
      Rcpp::XPtr<forestry> ptr(testFullForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeforestry,
        (Rboolean) TRUE
      );
      return ptr;
    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }

  } else {

    try {
      std::unique_ptr< std::vector< std::vector<double> > > featureDataRcpp (
          new std::vector< std::vector<double> >(
              Rcpp::as< std::vector< std::vector<double> > >(x)
          )
      );

      std::unique_ptr< std::vector<double> > outcomeDataRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(y)
          )
      );

      std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(catCols)
          )
      );

      std::unique_ptr< std::vector<size_t> > linearFeats (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(linCols)
          )
      );

      std::sort(linearFeats->begin(), linearFeats->end());

      std::unique_ptr< std::vector<double> > featureWeightsRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(featureWeights)
          )
      );

      std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(deepFeatureWeights)
          )
      );


      std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
          )
      );

      std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
          )
      );

      std::unique_ptr< std::vector<double> > observationWeightsRcpp (
          new std::vector<double>(
              Rcpp::as< std::vector<double> >(observationWeights)
          )
      );

      std::unique_ptr< std::vector<int> > monotoneConstraintsRcpp (
          new std::vector<int>(
              Rcpp::as< std::vector<int> >(monotonicConstraints)
          )
      );

      std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(groupMemberships)
          )
      );

      std::unique_ptr< std::vector<size_t> > symmetricIndicesRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(symmetricIndices)
          )
      );

      DataFrame* trainingData = new DataFrame(
          std::move(featureDataRcpp),
          std::move(outcomeDataRcpp),
          std::move(categoricalFeatureColsRcpp),
          std::move(linearFeats),
          (size_t) numRows,
          (size_t) numColumns,
          std::move(featureWeightsRcpp),
          std::move(featureWeightsVariablesRcpp),
          std::move(deepFeatureWeightsRcpp),
          std::move(deepFeatureWeightsVariablesRcpp),
          std::move(observationWeightsRcpp),
          std::move(monotoneConstraintsRcpp),
          std::move(groupMembershipsRcpp),
          (bool) monotoneAvg,
          std::move(symmetricIndicesRcpp)
      );

      forestry* testFullForest = new forestry(
        trainingData,
        (size_t) ntree,
        replace,
        (size_t) sampsize,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (double) minSplitGain,
        (size_t) maxDepth,
        (size_t) interactionDepth,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        (size_t) minTreesPerGroup,
        hasNas,
        linear,
        symmetric,
        (double) overfitPenalty,
        doubleTree
      );

      // delete(testFullForest);
      Rcpp::XPtr<forestry> ptr(testFullForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeforestry,
        (Rboolean) TRUE
      );
      return ptr;

    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }
  }
  return NULL;
}

// [[Rcpp::export]]
SEXP rcpp_cppMultilayerBuildInterface(
    Rcpp::List x,
    Rcpp::NumericVector y,
    Rcpp::NumericVector catCols,
    Rcpp::NumericVector linCols,
    int numRows,
    int numColumns,
    int ntree,
    int nrounds,
    double eta,
    bool replace,
    int sampsize,
    int mtry,
    double splitratio,
    bool OOBhonest,
    bool doubleBootstrap,
    int nodesizeSpl,
    int nodesizeAvg,
    int nodesizeStrictSpl,
    int nodesizeStrictAvg,
    double minSplitGain,
    int maxDepth,
    int seed,
    int nthread,
    bool verbose,
    bool middleSplit,
    int maxObs,
    Rcpp::NumericVector featureWeights,
    Rcpp::NumericVector featureWeightsVariables,
    Rcpp::NumericVector deepFeatureWeights,
    Rcpp::NumericVector deepFeatureWeightsVariables,
    Rcpp::NumericVector observationWeights,
    Rcpp::NumericVector monotonicConstraints,
    bool linear,
    double overfitPenalty,
    bool doubleTree,
    bool existing_dataframe_flag,
    SEXP existing_dataframe
){

  if (existing_dataframe_flag) {

    try {
      Rcpp::XPtr< DataFrame > trainingData(existing_dataframe) ;

      multilayerForestry* testMultiForest = new multilayerForestry(
        trainingData,
        (size_t) ntree,
        (size_t) nrounds,
        (double) eta,
        replace,
        (size_t) sampsize,
        splitratio,
        OOBhonest,
        doubleBootstrap,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (double) minSplitGain,
        (size_t) maxDepth,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        linear,
        (double) overfitPenalty,
        doubleTree
      );

      // delete(testFullForest);
      Rcpp::XPtr<multilayerForestry> ptr(testMultiForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeMultilayerForestry,
        (Rboolean) TRUE
      );
      return ptr;
    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }
  } else {
    Rcpp::Rcout << "Issue with Multilayer DataFrame.";
  }

  return NULL;
}

// [[Rcpp::export]]
Rcpp::List rcpp_cppPredictInterface(
  SEXP forest,
  Rcpp::List x,
  std::string aggregation,
  int seed,
  int nthread,
  bool exact,
  bool returnWeightMatrix,
  bool use_weights,
  bool use_hold_out_idx,
  Rcpp::NumericVector tree_weights,
  Rcpp::IntegerVector hold_out_idx,
  bool fillRidgeCoefficients
){
  try {

      Rcpp::XPtr <forestry> testFullForest(forest);

      std::vector <std::vector<double>> featureData =
              Rcpp::as < std::vector < std::vector < double > > > (x);

      std::unique_ptr <std::vector<double>> testForestPrediction;
      // We always initialize the weightMatrix. If the aggregation is weightMatrix
      // then we inialize the empty weight matrix
      arma::Mat<double> weightMatrix;
      arma::Mat<int> terminalNodes;
      arma::Mat<double> coefficients;

      if (returnWeightMatrix) {
          size_t nrow = featureData[0].size(); // number of features to be predicted
          size_t ncol = (*testFullForest).getNtrain(); // number of train data
          weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
          weightMatrix.zeros(nrow, ncol);  // set it all to 0
      }


      // Have to keep track of tree_weights
      std::vector <size_t> *testForestTreeWeights;
      std::vector <size_t> weights;

      // If using predict indices, set weights according to them
      if (use_hold_out_idx) {
          std::vector <size_t> holdOutIdxCpp = Rcpp::as < std::vector < size_t > > (hold_out_idx);

          for (auto &tree: *(testFullForest->getForest())) {
              bool discard_tree = false;
              std::unordered_set <size_t> hold_out_set(holdOutIdxCpp.begin(), holdOutIdxCpp.end());
              for (const auto &averaging_index: *(tree->getAveragingIndex())) {
                  if (hold_out_set.count(averaging_index)) {
                      discard_tree = true;
                      break;
                  }
              }
              // if Still haven't found any of them, search splitting set
              if (!discard_tree) {
                  for (const auto &splitting_index: *(tree->getSplittingIndex())) {
                      if (hold_out_set.count(splitting_index)) {
                          discard_tree = true;
                          break;
                      }
                  }
              }
              if (discard_tree) {
                  weights.push_back(0);
              } else {
                  weights.push_back(1);
              }
          } // End tree loop
          // Tell forest to use the weights
          use_weights = true;
      } else {
          // If we have weights we want to initialize them.
          weights = Rcpp::as < std::vector < size_t > > (tree_weights);
      }

      // Make ptr to weights
      testForestTreeWeights =
              new std::vector<size_t>(weights);


      size_t threads_to_use;
      if (nthread == 0) {
          threads_to_use = testFullForest->getNthread();
      } else {
          threads_to_use = (size_t) nthread;
      }

      if (fillRidgeCoefficients) {
          testForestPrediction = (*testFullForest).predict(&featureData,
                                                           returnWeightMatrix ? &weightMatrix : NULL,
                                                           NULL,
                                                           NULL,
                                                           seed,
                                                           threads_to_use,
                                                           exact,
                                                           use_weights,
                                                           use_weights ? testForestTreeWeights : NULL,
                                                           true);
      } else if (aggregation == "coefs") {
      size_t nrow = featureData[0].size();
      // Now we need the number of linear features + 1 for the intercept
      size_t ncol = (*testFullForest).getTrainingData()->getLinObsData(0).size() + 1;
      //Set coefficients to be zero
      coefficients.resize(nrow, ncol);
      coefficients.zeros(nrow, ncol);

      testForestPrediction = (*testFullForest).predict(&featureData,
                                                       NULL,
                                                       &coefficients,
                                                       NULL,
                                                       seed,
                                                       threads_to_use,
                                                       false,
                                                       false,
                                                       NULL,
                                                       false);

    } else if (aggregation == "terminalNodes") {
      // In this case, we return both the terminal nodes, and the weightMatrix
      size_t nrow = featureData[0].size(); // number of features to be predicted
      size_t ncol = (*testFullForest).getNtrain(); // number of train data
      weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
      weightMatrix.zeros(nrow, ncol);  // set it all to 0

      // Don't make sparse matrix in C,
      // get indices for each observation/tree combo,
      // then parse the sparse form in R
      ncol = (*testFullForest).getNtree();  // Total nodes across the forest
      nrow = featureData[1].size()+1;   // Total feature.new observations
      // Bottom row is the total node count/tree


      terminalNodes.resize(nrow, ncol);
      terminalNodes.zeros(nrow, ncol);
      // The idea is that, if the weightMatrix is point to NULL it won't be
      // be updated, but otherwise it will be updated:
      testForestPrediction = (*testFullForest).predict(&featureData,
                                                       &weightMatrix,
                                                       NULL,
                                                       &terminalNodes,
                                                       seed,
                                                       threads_to_use,
                                                       exact,
                                                       false,
                                                       NULL,
                                                       false);
    } else {
      // If the weights are zero, we just return NaN's
      if (use_weights &&
          (std::accumulate(testForestTreeWeights->begin(), testForestTreeWeights->end(), 0) == 0)) {

        testForestPrediction = std::unique_ptr< std::vector<double> >(
          new std::vector<double>(featureData[0].size(), std::numeric_limits<double>::quiet_NaN())
        );
      } else {
        testForestPrediction = (*testFullForest).predict(&featureData,
                                returnWeightMatrix ? &weightMatrix : NULL,
                                NULL,
                                NULL,
                                seed,
                                threads_to_use,
                                exact,
                                use_weights,
                                use_weights ? testForestTreeWeights : NULL,
                                false);
      }
    }

    std::vector<double>* testForestPrediction_ =
      new std::vector<double>(*testForestPrediction.get());

    Rcpp::NumericVector predictions = Rcpp::wrap(*testForestPrediction_);

    delete testForestPrediction_;
    delete testForestTreeWeights;

    return Rcpp::List::create(Rcpp::Named("predictions") = predictions,
                              Rcpp::Named("weightMatrix") = weightMatrix,
                              Rcpp::Named("terminalNodes") = terminalNodes,
                              Rcpp::Named("coef") = coefficients);

    // return output;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
Rcpp::List rcpp_cppMultilayerPredictInterface(
    SEXP multilayerForest,
    Rcpp::List x,
    std::string aggregation,
    int seed,
    int nthread,
    bool exact
){
  try {

    Rcpp::XPtr< multilayerForestry > testMultiForest(multilayerForest) ;

    std::vector< std::vector<double> > featureData =
      Rcpp::as< std::vector< std::vector<double> > >(x);

    std::unique_ptr< std::vector<double> > testMultiForestPrediction;

    size_t threads_to_use;
    if (nthread == 0) {
      threads_to_use = testMultiForest->getNthread();
    } else {
      threads_to_use = (size_t) nthread;
    }
    // We always initialize the weightMatrix. If the aggregation is weightMatrix
    // then we inialize the empty weight matrix
    arma::Mat<double> weightMatrix;
    if(aggregation == "weightMatrix") {
      size_t nrow = featureData[0].size(); // number of features to be predicted
      size_t ncol = (*testMultiForest).getNtrain(); // number of train data
      weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
      weightMatrix.zeros(nrow, ncol);// set it all to 0

      // The idea is that, if the weightMatrix is point to NULL it won't be
      // be updated, but otherwise it will be updated:
      testMultiForestPrediction = (*testMultiForest).predict(&featureData,
                                                             &weightMatrix,
                                                             seed,
                                                             threads_to_use,
                                                             exact);
    } else {
      testMultiForestPrediction = (*testMultiForest).predict(&featureData,
                                                             NULL,
                                                             seed,
                                                             threads_to_use,
                                                             exact);
    }

    std::vector<double>* testMultiForestPrediction_ =
      new std::vector<double>(*testMultiForestPrediction.get());

    Rcpp::NumericVector predictions = Rcpp::wrap(*testMultiForestPrediction_);

    return Rcpp::List::create(Rcpp::Named("predictions") = predictions,
                              Rcpp::Named("weightMatrix") = weightMatrix);

    // return output;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}


// [[Rcpp::export]]
double rcpp_OBBPredictInterface(
    SEXP forest
){

  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    double OOBError = (*testFullForest).getOOBError();
    return OOBError;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na();
}

// [[Rcpp::export]]
Rcpp::List rcpp_OBBPredictionsInterface(
    SEXP forest,
    Rcpp::List x,
    bool existing_df,
    bool doubleOOB,
    bool returnWeightMatrix,
    bool exact,
    bool use_training_idx,
    Rcpp::IntegerVector training_idx
){
  // Then we predict with the feature.new data
  if (existing_df) {
    std::vector< std::vector<double> > featureData =
      Rcpp::as< std::vector< std::vector<double> > >(x);

    std::vector<size_t> training_idx_cpp;
    if (use_training_idx){
        training_idx_cpp = Rcpp::as< std::vector<size_t> >(training_idx);
    }

    try {
      Rcpp::XPtr< forestry > testFullForest(forest) ;

      arma::Mat<double> weightMatrix;

      if (returnWeightMatrix) {
        size_t nrow = use_training_idx ? training_idx.size() : (*testFullForest).getNtrain(); // number of features to be predicted
        size_t ncol = (*testFullForest).getNtrain(); // number of train data
        weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
        weightMatrix.zeros(nrow, ncol);// set it all to 0

        std::vector<double> OOBpreds = (*testFullForest).predictOOB(&featureData,
                                        &weightMatrix,
                                        doubleOOB,
                                        exact,
                                        training_idx_cpp);
        Rcpp::NumericVector wrapped_preds = Rcpp::wrap(OOBpreds);

        return Rcpp::List::create(Rcpp::Named("predictions") = wrapped_preds,
                                  Rcpp::Named("weightMatrix") = weightMatrix);
      } else {
        // If we don't need weightMatrix, don't return it
        std::vector<double> OOBpreds = (*testFullForest).predictOOB(&featureData,
                                        NULL,
                                        doubleOOB,
                                        exact,
                                        training_idx_cpp);
        Rcpp::NumericVector wrapped_preds = Rcpp::wrap(OOBpreds);

        return Rcpp::List::create(Rcpp::Named("predictions") = wrapped_preds);
      }

    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }

  // Otherwise we predict with just the in sample data
  } else {
    try {
      Rcpp::XPtr< forestry > testFullForest(forest) ;
      std::vector<double> OOBpreds = (*testFullForest).getOOBpreds(doubleOOB);
      Rcpp::NumericVector wrapped_preds = Rcpp::wrap(OOBpreds);
      return Rcpp::List::create(Rcpp::Named("predictions") = wrapped_preds);
    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }
  }

  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
Rcpp::List rcpp_VariableImportanceInterface(
  SEXP forest
){

  try {
    Rcpp::XPtr< forestry > testFullForest(forest);
    std::vector<double> variableImportances = testFullForest->getVariableImportance();
    Rcpp::NumericVector importances = Rcpp::wrap(variableImportances);
    return Rcpp::List::create(importances);
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na();
}


// [[Rcpp::export]]
double rcpp_getObservationSizeInterface(
    SEXP df
){

  try {
    Rcpp::XPtr< DataFrame > trainingData(df) ;
    double nrows = (double) (*trainingData).getNumRows();
    return nrows;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na();
}


// [[Rcpp::export]]
void rcpp_AddTreeInterface(
    SEXP forest,
    int ntree
){
  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    (*testFullForest).addTrees(ntree);
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
}

// [[Rcpp::export]]
Rcpp::List rcpp_CppToR_translator(
    SEXP forest
){
  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    std::unique_ptr< std::vector<tree_info> > forest_dta(
      new std::vector<tree_info>
    );
    (*testFullForest).fillinTreeInfo(forest_dta);

    //   Print statements for debugging
    // std::cout << "hello\n";
    // std::cout.flush();

    // Return the lis of list. For each tree an element in the first list:
    Rcpp::List list_to_return;

    for(size_t i=0; i!=forest_dta->size(); i++){
      Rcpp::IntegerVector var_id = Rcpp::wrap(((*forest_dta)[i]).var_id);

      // std::cout << "var_id\n";
      // std::cout.flush();

      Rcpp::NumericVector split_val = Rcpp::wrap(((*forest_dta)[i]).split_val);

      // std::cout << "split_val\n";
      // std::cout.flush();


      Rcpp::IntegerVector averagingSampleIndex =
	      Rcpp::wrap(((*forest_dta)[i]).averagingSampleIndex);

      // std::cout << "averagingSampleIndex\n";
      // std::cout.flush();

      Rcpp::IntegerVector splittingSampleIndex =
	      Rcpp::wrap(((*forest_dta)[i]).splittingSampleIndex);

      // std::cout << "splittingSampleIndex\n";
      // std::cout.flush();

      Rcpp::IntegerVector naLeftCounts =
        Rcpp::wrap(((*forest_dta)[i]).naLeftCount);

      Rcpp::IntegerVector naRightCounts =
        Rcpp::wrap(((*forest_dta)[i]).naRightCount);

      Rcpp::NumericVector predictWeights =
              Rcpp::wrap(((*forest_dta)[i]).values);


        Rcpp::List list_i =
        Rcpp::List::create(
			   Rcpp::Named("var_id") = var_id,
			   Rcpp::Named("split_val") = split_val,
			   Rcpp::Named("averagingSampleIndex") = averagingSampleIndex,
			   Rcpp::Named("splittingSampleIndex") = splittingSampleIndex,
			   Rcpp::Named("naLeftCounts") = naLeftCounts,
			   Rcpp::Named("naRightCounts") = naRightCounts,
			   Rcpp::Named("seed") = (*forest_dta)[i].seed, // Add the seeds to the list we return
               Rcpp::Named("weights") = predictWeights
        );

      // std::cout << "finished list\n";
      // std::cout.flush();

      list_to_return.push_back(list_i);
    }

    // std::cout << "hello1\n";
    // std::cout.flush();



    return list_to_return;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
Rcpp::NumericVector rcpp_gammas_translator(
    SEXP multilayerForest
) {
  try{
    Rcpp::XPtr< multilayerForestry > testFullForest(multilayerForest);

    // Read off the gammas and return as a numeric vector
    Rcpp::NumericVector ret_gammas = Rcpp::wrap(testFullForest->getGammas());
    return ret_gammas;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na();
}

// [[Rcpp::export]]
Rcpp::List rcpp_residuals_translator(
    SEXP multilayerForest
) {
  try{
    Rcpp::XPtr< multilayerForestry > testFullForest(multilayerForest);

    // Read off the gammas and return as a numeric vector
    Rcpp::List ret_residuals;
    std::vector< forestry* >* forests = testFullForest->getMultilayerForests();
    for (size_t i = 0; i < testFullForest->getMultilayerForests()->size(); i++) {
      ret_residuals.push_back(Rcpp::wrap(*((*forests)[i]->getTrainingData()->getOutcomeData())));
    }

    return ret_residuals;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
Rcpp::List rcpp_multilayer_CppToR_translator(
    SEXP multilayerForest
){
  try {
    // std::cout << "Get the ptr \n";
    // std::cout.flush();
    Rcpp::XPtr< multilayerForestry > testFullForest(multilayerForest);

    // std::cout << "Make ptr \n";
    // std::cout.flush();
    std::vector< std::unique_ptr< std::vector<tree_info> > > forest_dta;

    // std::cout << "unit ptr \n";
    // std::cout.flush();
    // Now I make a vector of ptr's to each forest's trees
    for (size_t j = 0; j < testFullForest->getMultilayerForests()->size(); j++) {
      forest_dta.push_back(std::unique_ptr<std::vector<tree_info>> (new std::vector<tree_info>));
    }
    // std::cout << "fill in tree data \n";
    // std::cout.flush();
    // Now fill in the tree data for each
    for (size_t j = 0; j < testFullForest->getMultilayerForests()->size(); j++) {
      (*testFullForest->getMultilayerForests())[j]->fillinTreeInfo(forest_dta[j]);
    }
    // std::cout << "read in tree data \n";
    // std::cout.flush();
    // Return the list of list. For each tree an element in the first list:
    Rcpp::List list_to_return;

    for(size_t j=0; j  < forest_dta.size(); j++) {
      Rcpp::List list_to_return_j;

      for (size_t i = 0; i < forest_dta[j]->size(); i++ ) {
        Rcpp::IntegerVector var_id = Rcpp::wrap((*(forest_dta[j]))[i].var_id);

        // std::cout << "var_id\n";
        // std::cout.flush();

        Rcpp::NumericVector split_val = Rcpp::wrap((*(forest_dta[j]))[i].split_val);
        // std::cout << "split_val\n";
        // std::cout.flush();

        Rcpp::IntegerVector averagingSampleIndex =
          Rcpp::wrap((*(forest_dta[j]))[i].averagingSampleIndex);
        // std::cout << "averagingSampleIndex\n";
        // std::cout.flush();

        Rcpp::IntegerVector splittingSampleIndex =
          Rcpp::wrap((*(forest_dta[j]))[i].splittingSampleIndex);
        // std::cout << "splittingSampleIndex\n";
        // std::cout.flush();

        Rcpp::IntegerVector naLeftCounts =
          Rcpp::wrap((*(forest_dta[j]))[i].naLeftCount);

        Rcpp::IntegerVector naRightCounts =
          Rcpp::wrap((*(forest_dta[j]))[i].naRightCount);

        Rcpp::NumericVector predictWeights =
                  Rcpp::wrap((*(forest_dta[j]))[i].values);

        Rcpp::List list_i =
          Rcpp::List::create(
            Rcpp::Named("var_id") = var_id,
            Rcpp::Named("split_val") = split_val,
            Rcpp::Named("averagingSampleIndex") = averagingSampleIndex,
            Rcpp::Named("splittingSampleIndex") = splittingSampleIndex,
            Rcpp::Named("naLeftCounts") = naLeftCounts,
            Rcpp::Named("naRightCounts") = naRightCounts,
            Rcpp::Named("seed") = (*(forest_dta[j]))[i].seed,
            Rcpp::Named("weights") = predictWeights
          );

        // std::cout << "finished list\n";
        // std::cout.flush();

        list_to_return_j.push_back(list_i);

        // std::cout << i << "pushed list\n";
        // std::cout.flush();
      }

      // std::cout << "pushing final list\n";
      // std::cout.flush();

      list_to_return.push_back(list_to_return_j);
    }

    // std::cout << "hello1\n";
    // std::cout.flush();

    return list_to_return;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::List::create(NA_REAL);
}

// [[Rcpp::export]]
Rcpp::List rcpp_reconstructree(
  Rcpp::List x,
  Rcpp::NumericVector y,
  Rcpp::NumericVector catCols,
  Rcpp::NumericVector linCols,
  int numRows,
  int numColumns,
  Rcpp::List R_forest,
  bool replace,
  int sampsize,
  double splitratio,
  bool OOBhonest,
  bool doubleBootstrap,
  int mtry,
  int nodesizeSpl,
  int nodesizeAvg,
  int nodesizeStrictSpl,
  int nodesizeStrictAvg,
  double minSplitGain,
  int maxDepth,
  int interactionDepth,
  int seed,
  int nthread,
  bool verbose,
  bool middleSplit,
  int maxObs,
  int minTreesPerGroup,
  Rcpp::NumericVector featureWeights,
  Rcpp::NumericVector featureWeightsVariables,
  Rcpp::NumericVector deepFeatureWeights,
  Rcpp::NumericVector deepFeatureWeightsVariables,
  Rcpp::NumericVector observationWeights,
  Rcpp::NumericVector monotonicConstraints,
  Rcpp::NumericVector groupMemberships,
  bool monotoneAvg,
  int symmetricIndex,
  bool hasNas,
  bool linear,
  Rcpp::NumericVector symmetric,
  double overfitPenalty,
  bool doubleTree
){

  // Decode the R_forest data and create appropriate pointers to pointers:
  std::unique_ptr< std::vector< std::vector<int> > > var_ids(
      new std::vector< std::vector<int> >
  );
  std::unique_ptr< std::vector< std::vector<double> > > split_vals(
      new  std::vector< std::vector<double> >
  );
  std::unique_ptr< std::vector< std::vector<int> > > naLeftCounts(
      new std::vector< std::vector<int> >
  );
  std::unique_ptr< std::vector< std::vector<int> > > naRightCounts(
      new std::vector< std::vector<int> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > averagingSampleIndex(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > splittingSampleIndex(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector<unsigned int> > tree_seeds(
      new std::vector<unsigned int>
  );
  std::unique_ptr< std::vector< std::vector<double> > > predictWeights(
          new  std::vector< std::vector<double> >
  );

    // Reserve space for each of the vectors equal to R_forest.size()
  var_ids->reserve(R_forest.size());
  split_vals->reserve(R_forest.size());
  averagingSampleIndex->reserve(R_forest.size());
  splittingSampleIndex->reserve(R_forest.size());
  naLeftCounts->reserve(R_forest.size());
  naRightCounts->reserve(R_forest.size());
  tree_seeds->reserve(R_forest.size());
  predictWeights->reserve(R_forest.size());


  // Now actually populate the vectors
  for(int i=0; i!=R_forest.size(); i++){
    var_ids->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[0])
      );
    split_vals->push_back(
        Rcpp::as< std::vector<double> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[1])
      );
    averagingSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[2])
      );
    splittingSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[3])
      );
    naLeftCounts->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[4])
    );
    naRightCounts->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[5])
    );
    tree_seeds->push_back(
        Rcpp::as< unsigned int > ((Rcpp::as<Rcpp::List>(R_forest[i]))[6])
    );
    predictWeights->push_back(
            Rcpp::as< std::vector<double> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[7])
    );
  }

  // Decode catCols and R_forest
  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(catCols)
      )
  ); // contains the col indices of categorical features.

  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp_copy(
      new std::vector<size_t>
  );

  for(size_t i=0; i<(*categoricalFeatureColsRcpp).size(); i++){
    (*categoricalFeatureColsRcpp_copy).push_back(
        (*categoricalFeatureColsRcpp)[i]);
  }

  std::unique_ptr<std::vector< std::vector<double> > > featureDataRcpp (
      new std::vector< std::vector<double> >(
          Rcpp::as< std::vector< std::vector<double> > >(x)
      )
  );

  std::unique_ptr< std::vector<double> > outcomeDataRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(y)
      )
  );

  std::unique_ptr< std::vector<size_t> > linearFeats (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(linCols)
      )
  );

  std::sort(linearFeats->begin(), linearFeats->end());

  std::unique_ptr< std::vector<double> > featureWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(featureWeights)
      )
  );

  std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
      )
  );

  std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(deepFeatureWeights)
      )
  );
  std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
      )
  );
  std::unique_ptr< std::vector<double> > observationWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(observationWeights)
      )
  );
  std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp (
      new std::vector<int>(
          Rcpp::as< std::vector<int> >(monotonicConstraints)
      )
  );
  std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(groupMemberships)
      )
  );

  std::unique_ptr< std::vector<size_t> > symmetricIndicesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(groupMemberships)
      )
  );

  DataFrame* trainingData = new DataFrame(
    std::move(featureDataRcpp),
    std::move(outcomeDataRcpp),
    std::move(categoricalFeatureColsRcpp),
    std::move(linearFeats),
    (size_t) numRows,
    (size_t) numColumns,
    std::move(featureWeightsRcpp),
    std::move(featureWeightsVariablesRcpp),
    std::move(deepFeatureWeightsRcpp),
    std::move(deepFeatureWeightsVariablesRcpp),
    std::move(observationWeightsRcpp),
    std::move(monotonicConstraintsRcpp),
    std::move(groupMembershipsRcpp),
    (bool) monotoneAvg,
    std::move(symmetricIndicesRcpp)
  );

  forestry* testFullForest = new forestry(
    (DataFrame*) trainingData,
    (int) 0,
    (bool) replace,
    (int) sampsize,
    (double) splitratio,
    (bool) OOBhonest,
    (bool) doubleBootstrap,
    (int) mtry,
    (int) nodesizeSpl,
    (int) nodesizeAvg,
    (int) nodesizeStrictSpl,
    (int) nodesizeStrictAvg,
    (double) minSplitGain,
    (int) maxDepth,
    (int) interactionDepth,
    (unsigned int) seed,
    (int) nthread,
    (bool) verbose,
    (bool) middleSplit,
    (int) maxObs,
    (size_t) minTreesPerGroup,
    (bool) hasNas,
    (bool) linear,
    (bool) symmetric,
    (double) overfitPenalty,
    doubleTree
  );

  testFullForest->reconstructTrees(categoricalFeatureColsRcpp_copy,
                                   tree_seeds,
                                   var_ids,
                                   split_vals,
                                   naLeftCounts,
                                   naRightCounts,
                                   averagingSampleIndex,
                                   splittingSampleIndex,
                                   predictWeights
                                   );


  // delete(testFullForest);
  Rcpp::XPtr<forestry> ptr(testFullForest, true);
  R_RegisterCFinalizerEx(
    ptr,
    (R_CFinalizer_t) freeforestry,
    (Rboolean) TRUE
  );
  Rcpp::XPtr<DataFrame> df_ptr(trainingData, true) ;
  return Rcpp::List::create(Rcpp::Named("forest_ptr") = ptr,
                            Rcpp::Named("data_frame_ptr") = df_ptr);
}

// [[Rcpp::export]]
Rcpp::List rcpp_reconstruct_forests(
    Rcpp::List x,
    Rcpp::NumericVector y,
    Rcpp::NumericVector catCols,
    Rcpp::NumericVector linCols,
    int numRows,
    int numColumns,
    Rcpp::List R_forests,
    Rcpp::List R_residuals,
    int nrounds,
    double eta,
    bool replace,
    int sampsize,
    double splitratio,
    bool OOBhonest,
    bool doubleBootstrap,
    int mtry,
    int nodesizeSpl,
    int nodesizeAvg,
    int nodesizeStrictSpl,
    int nodesizeStrictAvg,
    double minSplitGain,
    int maxDepth,
    int seed,
    int nthread,
    bool verbose,
    bool middleSplit,
    int maxObs,
    int minTreesPerGroup,
    Rcpp::NumericVector featureWeights,
    Rcpp::NumericVector featureWeightsVariables,
    Rcpp::NumericVector deepFeatureWeights,
    Rcpp::NumericVector deepFeatureWeightsVariables,
    Rcpp::NumericVector observationWeights,
    Rcpp::NumericVector monotonicConstraints,
    Rcpp::NumericVector groupMemberships,
    Rcpp::NumericVector gammas,
    bool monotoneAvg,
    int symmetricIndex,
    bool linear,
    Rcpp::NumericVector symmetric,
    double overfitPenalty,
    bool doubleTree
){

  // Decode the R_forest data and create appropriate pointers to pointers:
  std::vector< std::unique_ptr< std::vector< std::vector<int> > > > var_ids;
  std::vector< std::unique_ptr< std::vector< std::vector<double> > > > split_vals;
  std::vector< std::unique_ptr< std::vector< std::vector<int> > > > naLeftCounts;
  std::vector< std::unique_ptr< std::vector< std::vector<int> > > > naRightCounts;
  std::vector< std::unique_ptr< std::vector< std::vector<size_t> > > > averagingSampleIndex;
  std::vector< std::unique_ptr< std::vector< std::vector<size_t> > > > splittingSampleIndex;
  std::vector< std::unique_ptr< std::vector<unsigned int> > > tree_seeds;
  std::vector< std::unique_ptr< std::vector< std::vector<double> > > > weights;

  std::vector< forestry* > multilayerForests;
  // Now we need to iterate through the length of number forests, and for each
  // forest, the number of trees

  // std::cout << "Made it into c++ \n";
  // std::cout.flush();
  for (size_t j = 0; j < (size_t) R_forests.size(); j++) {

      std::vector< std::vector<int> > cur_var_ids;
      std::vector< std::vector<double> > cur_split_vals;
      std::vector< std::vector<int> > cur_naLeftCounts;
      std::vector< std::vector<int> > cur_naRightCounts;
      std::vector< std::vector<size_t> > cur_averagingSampleIndex;
      std::vector< std::vector<size_t> > cur_splittingSampleIndex;
      std::vector< std::vector<double> > cur_weights;
      std::vector< unsigned int > cur_tree_seeds;

    // Now for the current forest, we iterate through and build the trees
    for(size_t i = 0; i < (size_t) Rcpp::as<Rcpp::List>(R_forests[j]).size(); i++){
      // std::cout << "casting the lists \n";
      // std::cout.flush();

      cur_var_ids.push_back(
          Rcpp::as< std::vector<int> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[0])
      );

      cur_split_vals.push_back(
          Rcpp::as< std::vector<double> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[1])
      );

      cur_averagingSampleIndex.push_back(
        Rcpp::as< std::vector<size_t> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[2])
      );

      cur_splittingSampleIndex.push_back(
        Rcpp::as< std::vector<size_t> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[3])
      );

      cur_naLeftCounts.push_back(
        Rcpp::as< std::vector<int> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[4])
      );

      cur_naRightCounts.push_back(
        Rcpp::as< std::vector<int> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[5])
      );

      cur_tree_seeds.push_back(
        Rcpp::as< unsigned int > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[6])
      );

      cur_weights.push_back(
              Rcpp::as< std::vector<double> > (Rcpp::as<Rcpp::List>(Rcpp::as<Rcpp::List>(R_forests[j])[i])[7])
      );
    }
    // Now the cur vectors hold the info for each tree, we have to
    // add this info to the vector of forests
    var_ids.push_back(std::unique_ptr< std::vector< std::vector<int> > >(
        new std::vector< std::vector<int> >(cur_var_ids)
    ));

    split_vals.push_back(std::unique_ptr< std::vector< std::vector<double> > >(
        new std::vector< std::vector<double> >(cur_split_vals)
    ));

    naLeftCounts.push_back(std::unique_ptr< std::vector< std::vector<int> > >(
        new std::vector< std::vector<int> >(cur_naLeftCounts)
    ));

    naRightCounts.push_back(std::unique_ptr< std::vector< std::vector<int> > >(
        new std::vector< std::vector<int> >(cur_naRightCounts)
    ));

    averagingSampleIndex.push_back(std::unique_ptr< std::vector< std::vector<size_t> > >(
        new std::vector< std::vector<size_t> >(cur_averagingSampleIndex)
    ));

    splittingSampleIndex.push_back(std::unique_ptr< std::vector< std::vector<size_t> > >(
        new std::vector< std::vector<size_t> >(cur_splittingSampleIndex)
    ));

    tree_seeds.push_back(std::unique_ptr< std::vector<unsigned int> >(
        new std::vector< unsigned int >(cur_tree_seeds)
    ));
    weights.push_back(std::unique_ptr< std::vector< std::vector<double> > >(
            new std::vector< std::vector<double> >(cur_weights)
    ));

    // Decode catCols and R_forest
    std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(catCols)
        )
    ); // contains the col indices of categorical features.


    std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp_copy(
        new std::vector<size_t>
    );

    for(size_t i=0; i<(*categoricalFeatureColsRcpp).size(); i++){
      (*categoricalFeatureColsRcpp_copy).push_back(
          (*categoricalFeatureColsRcpp)[i]);
    }

    std::unique_ptr<std::vector< std::vector<double> > > featureDataRcpp (
        new std::vector< std::vector<double> >(
            Rcpp::as< std::vector< std::vector<double> > >(x)
        )
    );

    std::unique_ptr< std::vector<double> > outcomeDataRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(R_residuals.at(j))
        )
    );

    std::unique_ptr< std::vector<size_t> > linearFeats (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(linCols)
        )
    );

    std::sort(linearFeats->begin(), linearFeats->end());

    std::unique_ptr< std::vector<double> > featureWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(featureWeights)
        )
    );

    std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
        )
    );

    std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(deepFeatureWeights)
        )
    );
    std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
        )
    );
    std::unique_ptr< std::vector<double> > observationWeightsRcpp (
        new std::vector<double>(
            Rcpp::as< std::vector<double> >(observationWeights)
        )
    );
    std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp (
        new std::vector<int>(
            Rcpp::as< std::vector<int> >(monotonicConstraints)
        )
    );
    std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(groupMemberships)
        )
    );

    std::unique_ptr< std::vector<size_t> > symmetricIndicesRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(groupMemberships)
        )
    );


    DataFrame* trainingData = new DataFrame(
      std::move(featureDataRcpp),
      std::move(outcomeDataRcpp),
      std::move(categoricalFeatureColsRcpp),
      std::move(linearFeats),
      (size_t) numRows,
      (size_t) numColumns,
      std::move(featureWeightsRcpp),
      std::move(featureWeightsVariablesRcpp),
      std::move(deepFeatureWeightsRcpp),
      std::move(deepFeatureWeightsVariablesRcpp),
      std::move(observationWeightsRcpp),
      std::move(monotonicConstraintsRcpp),
      std::move(groupMembershipsRcpp),
      (bool) monotoneAvg,
      std::move(symmetricIndicesRcpp)
    );

    // std::cout << "Making a forest \n";
    // std::cout.flush();

    forestry* testFullForest = new forestry(
      (DataFrame*) trainingData,
      (int) 0,
      (bool) replace,
      (int) sampsize,
      (double) splitratio,
      (bool) OOBhonest,
      (bool) doubleBootstrap,
      (int) mtry,
      (int) nodesizeSpl,
      (int) nodesizeAvg,
      (int) nodesizeStrictSpl,
      (int) nodesizeStrictAvg,
      (double) minSplitGain,
      (int) maxDepth,
      (int) maxDepth,
      (unsigned int) seed,
      (int) nthread,
      (bool) verbose,
      (bool) middleSplit,
      (int) maxObs,
      (size_t) minTreesPerGroup,
      false,
      (bool) linear,
      (bool) symmetric,
      (double) overfitPenalty,
      doubleTree
    );

    // std::cout << "RECONSTRUCT a forest \n";
    // std::cout.flush();
    // Reconstruct the jth forest with its tree info
    testFullForest->reconstructTrees(categoricalFeatureColsRcpp_copy,
                                     tree_seeds[j],
                                     var_ids[j],
                                     split_vals[j],
                                     naLeftCounts[j],
                                     naRightCounts[j],
                                     averagingSampleIndex[j],
                                     splittingSampleIndex[j],
                                     weights[j]);

    // Push back the jth forest to the vector of forests
    multilayerForests.push_back(testFullForest);

  }

  // Now we want to get the training data to add to the copies
  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(catCols)
      )
  ); // contains the col indices of categorical features.

  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp_copy(
      new std::vector<size_t>
  );

  for(size_t i=0; i<(*categoricalFeatureColsRcpp).size(); i++){
    (*categoricalFeatureColsRcpp_copy).push_back(
        (*categoricalFeatureColsRcpp)[i]);
  }

  std::unique_ptr<std::vector< std::vector<double> > > featureDataRcpp (
      new std::vector< std::vector<double> >(
          Rcpp::as< std::vector< std::vector<double> > >(x)
      )
  );

  std::unique_ptr< std::vector<double> > outcomeDataRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(y)
      )
  );

  std::unique_ptr< std::vector<size_t> > linearFeats (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(linCols)
      )
  );

  std::sort(linearFeats->begin(), linearFeats->end());
  std::unique_ptr< std::vector<double> > featureWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(featureWeights)
      )
  );

  std::unique_ptr< std::vector<size_t> > featureWeightsVariablesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(featureWeightsVariables)
      )
  );

  std::unique_ptr< std::vector<double> > deepFeatureWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(deepFeatureWeights)
      )
  );

  std::unique_ptr< std::vector<size_t> > deepFeatureWeightsVariablesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(deepFeatureWeightsVariables)
      )
  );

  std::unique_ptr< std::vector<double> > observationWeightsRcpp (
      new std::vector<double>(
          Rcpp::as< std::vector<double> >(observationWeights)
      )
  );

  std::unique_ptr< std::vector<int> > monotonicConstraintsRcpp (
      new std::vector<int>(
          Rcpp::as< std::vector<int> >(monotonicConstraints)
      )
  );
  std::unique_ptr< std::vector<size_t> > groupMembershipsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(groupMemberships)
      )
  );
  std::unique_ptr< std::vector<size_t> > symmetricIndicesRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(groupMemberships)
      )
  );

  DataFrame* trainingData = new DataFrame(
    std::move(featureDataRcpp),
    std::move(outcomeDataRcpp),
    std::move(categoricalFeatureColsRcpp),
    std::move(linearFeats),
    (size_t) numRows,
    (size_t) numColumns,
    std::move(featureWeightsRcpp),
    std::move(featureWeightsVariablesRcpp),
    std::move(deepFeatureWeightsRcpp),
    std::move(deepFeatureWeightsVariablesRcpp),
    std::move(observationWeightsRcpp),
    std::move(monotonicConstraintsRcpp),
    std::move(groupMembershipsRcpp),
    (bool) monotoneAvg,
    std::move(symmetricIndicesRcpp)
  );


  // NOW We need to make a multilayer forestry object and
  // populate this with the vector of forests we have created
  // delete(testFullForest);
  multilayerForestry* fullMultilayer = new multilayerForestry(
    (DataFrame*) trainingData,
    (size_t) multilayerForests[1]->getNtree(),     // Set ntree = 0
    (size_t) 0,     // set nrounds = 0
    (double) eta,
    (bool) replace,
    (int) sampsize,
    (double) splitratio,
    (bool) OOBhonest,
    (bool) doubleBootstrap,
    (size_t) mtry,
    (size_t) nodesizeSpl,
    (size_t) nodesizeAvg,
    (size_t) nodesizeStrictSpl,
    (size_t) nodesizeStrictAvg,
    (double) minSplitGain,
    (size_t) maxDepth,
    (unsigned int) seed,
    (size_t) nthread,
    (bool) verbose,
    (bool) middleSplit,
    (size_t) maxObs,
    (bool) linear,
    (double) overfitPenalty,
    (bool) doubleTree
  );

  // Get the gammas
  std::vector<double> forest_gammas =
    Rcpp::as< std::vector<double> >(gammas);

  // Now I need to figure out how to pass the forests
  fullMultilayer->reconstructForests(multilayerForests,
                                     forest_gammas);


  Rcpp::XPtr<multilayerForestry> ptr(fullMultilayer, true);
  R_RegisterCFinalizerEx(
    ptr,
    (R_CFinalizer_t) freeMultilayerForestry,
    (Rboolean) TRUE
  );


  Rcpp::XPtr<DataFrame> df_ptr(trainingData, true) ;
  return Rcpp::List::create(Rcpp::Named("forest_ptr") = ptr,
                            Rcpp::Named("data_frame_ptr") = df_ptr);
}

// [[Rcpp::export]]
std::vector< std::vector<double> > rcpp_cppImputeInterface(
    SEXP forest,
    Rcpp::List x,
    int seed
){
  // There is code duplication with rcpp_cppPredictInterface here. Really the
  // predict member function should be refactored so that the boilerplate
  // happens inside it.
  Rcpp::XPtr< forestry > testFullForest(forest);
  std::vector< std::vector<double> > featureData =
    Rcpp::as< std::vector< std::vector<double> > >(x);

  std::unique_ptr< std::vector<double> > testForestPrediction;
  arma::Mat<double> weightMatrix;

  size_t nrow = featureData[0].size(); // number of features to be predicted
  size_t ncol = (*testFullForest).getNtrain(); // number of train data
  weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
  weightMatrix.zeros(nrow, ncol); // set it all to 0

  testForestPrediction = (*testFullForest).predict(&featureData,
                                                   &weightMatrix,
                                                   NULL,
                                                   NULL,
                                                   seed,
                                                   testFullForest->getNthread(),
                                                   false,
                                                   false,
                                                   NULL,
                                                   false);

  std::vector<double>* testForestPrediction_ =
    new std::vector<double>(*testForestPrediction.get());

  Rcpp::NumericVector predictions = Rcpp::wrap(*testForestPrediction_);
  arma::Mat<double> weightMatrixT = weightMatrix;

  // Take tranpose because we want to access by row and armadillo uses column
  // major ordering.
  arma::inplace_trans(weightMatrixT);

  std::vector<std::vector<double>>* imputedX = testFullForest->neighborhoodImpute(
    &featureData,
    &weightMatrixT
  );
  //auto returnX = Rcpp::as<Rcpp::NumericMatrix>(imputedX);
  return *imputedX;
  //return weightMatrixT;
}
