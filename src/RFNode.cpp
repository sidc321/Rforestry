// [[Rcpp::depends(RcppThread)]]
// [[Rcpp::plugins(cpp11)]]
#include "RFNode.h"
#include <RcppArmadillo.h>
#include <RcppThread.h>
#include <mutex>
#include <thread>
#include "utils.h"

std::mutex mutex_weightMatrix;


RFNode::RFNode():
  _splitFeature(0), _splitValue(0),_trinary(false), _predictWeight(std::numeric_limits<double>::quiet_NaN()),
  _leftChild(nullptr),_rightChild(nullptr),_naLeftCount(0), _naRightCount(0),
  _naDefaultDirection(0), _averageCount(0), _splitCount(0) {}

RFNode::~RFNode() {
  //  std::cout << "RFNode() destructor is called." << std::endl;
};

void RFNode::setLeafNode(
        size_t averagingSampleIndexSize,
        size_t splittingSampleIndexSize,
        size_t nodeId,
        bool trinary,
        std::vector<double> weights,
        double predictWeight
) {
  if (
          averagingSampleIndexSize == 0 &&
                  splittingSampleIndexSize == 0
  ) {
    throw std::runtime_error("Intend to create an empty node.");
  }
  // Give the ownership of the index pointer to the RFNode object
  this->_naLeftCount = 0;
  this->_naRightCount = 0;
  this->_nodeId = nodeId;
  this->_averageCount = averagingSampleIndexSize;
  this->_splitCount = splittingSampleIndexSize;

  // Set the prediction weight for the node
  this->_predictWeight = predictWeight;

  if (trinary) {
    this->_trinary = trinary;
    this->_weights = weights;
  }
}

void RFNode::setSplitNode(
  size_t splitFeature,
  double splitValue,
  std::unique_ptr< RFNode > leftChild,
  std::unique_ptr< RFNode > rightChild,
  bool trinary,
  size_t naLeftCount,
  size_t naCenterCount,
  size_t naRightCount,
  int naDefaultDirection
) {
  // Split node constructor
  _splitCount = 0;
  _splitFeature = splitFeature;
  _splitValue = splitValue;
  // Give the ownership of the child pointer to the RFNode object
  _leftChild = std::move(leftChild);
  _rightChild = std::move(rightChild);
  _naLeftCount = naLeftCount;
  _naRightCount = naRightCount;
  _naDefaultDirection = naDefaultDirection;
  _trinary = trinary;
  _nodeId = -1;
}

void RFNode::setRidgeCoefficients(
        std::vector<size_t>* averagingIndices,
        DataFrame* trainingData,
        double lambda
) {

    //Observations to do regression with
    std::vector<size_t>* leafObs = averagingIndices;
    //Number of linear features in training data
    size_t dimension = (trainingData->getLinObsData((*leafObs)[0])).size();

    arma::Mat<double> x(leafObs->size(),
                        dimension + 1);
    arma::Mat<double> identity(dimension + 1,
                               dimension + 1);
    identity.eye();
    //Don't penalize intercept
    identity(dimension, dimension) = 0.0;
    std::vector<double> outcomePoints;
    std::vector<double> currentObservation;

    //Construct X and outcome vector
    for (size_t i = 0; i < leafObs->size(); i++) {
        currentObservation = trainingData->getLinObsData((*leafObs)[i]);
        currentObservation.push_back(1.0);
        x.row(i) = arma::conv_to<arma::Row<double> >::from(currentObservation);
        outcomePoints.push_back(trainingData->getOutcomePoint((*leafObs)[i]));
    }

    arma::Mat<double> y(outcomePoints.size(),
                        1);
    y.col(0) = arma::conv_to<arma::Col<double> >::from(outcomePoints);
    //Compute XtX + lambda * I * Y = C
    arma::Mat<double> coefficients = (x.t() * x +
                                      identity * lambda).i() * x.t() * y;

    this->_ridgeCoefficients = coefficients;
}

void RFNode::ridgePredict(
  std::vector<double> &outputPrediction,
  std::vector< std::vector<double> > &outputCoefficients,
  std::vector<size_t>* updateIndex,
  std::vector< std::vector<double> >* xNew,
  DataFrame* trainingData,
  double lambda
) {

  //Number of linear features in training data
  size_t dimension = (trainingData->getLinObsData(0)).size();
  // Pull the ridge regression coefficients
  arma::Mat<double> coefficients = getRidgeCoefficients();

  //Map xNew into Eigen matrix
  arma::Mat<double> xn(updateIndex->size(),
                       dimension + 1);

  size_t index = 0;
  for (std::vector<size_t>::iterator it = updateIndex->begin();
       it != updateIndex->end();
       ++it) {

    std::vector<double> newObservation;
    for (size_t i = 0; i < dimension; i++) {
      newObservation.push_back((*xNew)[i][*it]);
    }
    newObservation.push_back(1.0);

    xn.row(index) = arma::conv_to<arma::Row<double> >::from(newObservation);
    index++;
  }

  //Multiply xNew * coefficients = result
  arma::Mat<double> predictions = xn * coefficients;

  for (size_t i = 0; i < updateIndex->size(); i++) {
    outputPrediction[(*updateIndex)[i]] = predictions(i, 0);
  }

  // Want coefficients vector
  std::vector<double> c_vector =
    arma::conv_to< std::vector<double> >::from(coefficients.col(0));

  // If we want to update coefficients, update the vector as well
  if (!(outputCoefficients.empty())) {
    for (size_t k = 0; k < updateIndex->size(); k++) {
      outputCoefficients[(*updateIndex)[k]] = c_vector;
    }
  }
}

void RFNode::predict(
  std::vector<double> &outputPrediction,
  std::vector<int>* terminalNodes,
  std::vector< std::vector<double> > &outputCoefficients,
  std::vector<size_t>* updateIndex,
  std::vector<size_t>* predictionAveragingIndices,
  std::vector< std::vector<double> >* xNew,
  DataFrame* trainingData,
  arma::Mat<double>* weightMatrix,
  bool linear,
  bool naDirection,
  double lambda,
  unsigned int seed,
  size_t nodesizeStrictAvg,
  std::vector<size_t>* OOBIndex
) {

  // If the node is a leaf, aggregate all its averaging data samples
  if (is_leaf()) {

      if (linear) {

        //Use ridgePredict (fit linear model on leaf avging obs + evaluate it)
        ridgePredict(outputPrediction,
                     outputCoefficients,
                     updateIndex,
                     xNew,
                     trainingData,
                     lambda);

      } else {

        double predictedMean;
        // Calculate the mean of current node
        if (getAverageCount() == 0 && !getTrinary()) {
          predictedMean = std::numeric_limits<double>::quiet_NaN();
        } else if (getTrinary()) {
          predictedMean = 0;
        } else if (!std::isnan(getPredictWeight())) {
            predictedMean = getPredictWeight();
        } else {
          predictedMean = (*trainingData).partitionMean(getAveragingIndex());
        }



        // Give all updateIndex the mean of the node as prediction values
        for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it
        ) {
          if (getTrinary()) {
            // In this case, we test the feature value and use the correct
            // pseudo outcome by weight
            std::vector<bool> signs;
            for (size_t i = 0; i < (*trainingData->getSymmetricIndices()).size(); i++) {
              if ((*xNew)[(*trainingData->getSymmetricIndices())[i]][*it] > 0) {
                signs.push_back(true);
              } else {
                signs.push_back(false);
              }
            }

            // Pull the correct pseudo outcome
            size_t weight_idx = bin_to_idx(signs);

            outputPrediction[*it] =(this->getWeights())[weight_idx];

          } else {
            outputPrediction[*it] = predictedMean;
          }
        }
    }

    if (weightMatrix){
      // If weightMatrix is not a NULL pointer, then we want to update it,
      // because we have choosen aggregation = "weightmatrix".
      std::vector<size_t> idx_in_leaf =
                (*trainingData).get_all_row_idx(predictionAveragingIndices);


        // The following will lock the access to weightMatrix
      std::lock_guard<std::mutex> lock(mutex_weightMatrix);
      for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it ) {
        // Set the row which we update in the weightMatrix
        //
        size_t idx = *it;
        if (OOBIndex) {
          idx = (*OOBIndex)[*it];
        }

        for (size_t i = 0; i<idx_in_leaf.size(); i++) {
          (*weightMatrix)(idx, idx_in_leaf[i] - 1) +=
          (double) 1.0 / ((double) idx_in_leaf.size());
        }
      }
    }

    if (terminalNodes) {
      // If terminalNodes not a NULLPTR, set the terminal node for all X in this
      // leaf to be the leaf node_id
      size_t node_id = getNodeId();
      for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it
      ) {
        (*terminalNodes)[*it] = node_id;
      }
    }

  // If not a leaf then we need to separate the prediction tasks
  } else {

    // Separate prediction tasks to two children
    std::vector<size_t>* leftPartitionIndex = new std::vector<size_t>();
    std::vector<size_t>* rightPartitionIndex = new std::vector<size_t>();
    (*leftPartitionIndex).reserve((*updateIndex).size());
    (*rightPartitionIndex).reserve((*updateIndex).size());

    size_t naLeftCount = getNaLeftCount();
    size_t naRightCount = getNaRightCount();

    // Test if the splitting feature is categorical
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();
    bool categorical_split = (std::find(categorialCols.begin(),
                                       categorialCols.end(),
                                       getSplitFeature()) != categorialCols.end());

    // Initialize vectors of ratios in which we sample randomly to send
    // observations with NA values
    std::vector<size_t> naSampling;
    if (!categorical_split) {
      naSampling = {naLeftCount, naRightCount};
    }

    std::vector<size_t> naSampling_no_miss;
    if (!categorical_split) {
      naSampling_no_miss = {
        (*getLeftChild()).getAverageCountAlways(),
        (*getRightChild()).getAverageCountAlways()};
    }


    std::discrete_distribution<size_t> discrete_dist(
        naSampling.begin(), naSampling.end()
    );
    std::discrete_distribution<size_t> discrete_dist_nonmissing(
        naSampling_no_miss.begin(), naSampling_no_miss.end()
    );
    std::mt19937_64 random_number_generator;
    random_number_generator.seed(seed);

    // Test if the splitting feature is categorical
    if (
        categorical_split
    ){

      // If the splitting feature is categorical, split by (==) or (!=)
      for (
        std::vector<size_t>::iterator it = (*updateIndex).begin();
        it != (*updateIndex).end();
        ++it
      ) {
        double currentValue = (*xNew)[getSplitFeature()][*it];

        if (std::isnan(currentValue)) {
          size_t draw;

          // If naDirection is set to true, follow the split node's default
          // direction for NAs.
          if (naDirection) {
            draw = getNaDefaultDirection();
            // naDefaultDirection is -1 for left and 1 for right
            if (draw == -1) {
              draw = 0;
            }
          } else {
            // If we have a missing feature value, if no NAs were observed when
            // splitting send to left/right with probability proportional to
            // number of observations in left/right child node else send
            // right/left with probability in proportion to NA's which went
            // left/right when splitting
            if ((naLeftCount == 0) && (naRightCount == 0)) {
              draw = discrete_dist_nonmissing(random_number_generator);
            } else {
              draw = discrete_dist(random_number_generator);
            }
          }

          // Have to push to three indices when we have trinary splits
          if (draw == 0) {
            (*leftPartitionIndex).push_back(*it);
          } else {
            (*rightPartitionIndex).push_back(*it);
          }

        } else if (currentValue == getSplitValue()) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }

    } else {

      // For non-categorical, split to left (<) and right (>=) according to the
      // split value
      for (
        std::vector<size_t>::iterator it = (*updateIndex).begin();
        it != (*updateIndex).end();
        ++it
      ) {
        double currentValue = (*xNew)[getSplitFeature()][*it];

        if (std::isnan(currentValue)) {
          size_t draw;

          // If we have a missing feature value, if no NAs were observed when
          // splitting send to left/right with probability proportional to
          // number of observations in left/right child node else send
          // right/left with probability in proportion to NA's which went
          // left/right when splitting
          if (getTrinary()) {
            if ((naLeftCount == 0) && (naRightCount == 0)) {
              draw = discrete_dist_nonmissing(random_number_generator);
            } else {
              draw = discrete_dist(random_number_generator);
            }

            // Have to push to three indices when we have trinary splits
            if (draw == -1) {
              (*leftPartitionIndex).push_back(*it);
            } else if (draw == 1) {
              (*rightPartitionIndex).push_back(*it);
            }
          } else {
            // If naDirection is set to true, follow the split node's default
            // direction for NAs.
            if (naDirection) {
              draw = getNaDefaultDirection();
              // naDefaultDirection is -1 for left and 1 for right
              if (draw == -1) {
                draw = 0;
              }
            } else {
              if ((naLeftCount == 0) && (naRightCount == 0)) {
                draw = discrete_dist_nonmissing(random_number_generator);
              } else {
                draw = discrete_dist(random_number_generator);
              }
            }

            // Now push to index
            if (draw == 0) {
              (*leftPartitionIndex).push_back(*it);
            } else {
              (*rightPartitionIndex).push_back(*it);
            }
          }

        } else {
          // Check if the current split feature is a symmetric feature
          if (getTrinary() && (std::find(
              trainingData->getSymmetricIndices()->begin(),
              trainingData->getSymmetricIndices()->end(),
              getSplitFeature()) != trainingData->getSymmetricIndices()->end())) {

            // If this is a symmetric feature, have to use the absolute value
            // of the feature value
            if (std::fabs(currentValue) < getSplitValue()) {
              (*leftPartitionIndex).push_back(*it);
            } else {
              (*rightPartitionIndex).push_back(*it);
            }
          } else {
            // Run standard predictions
            if (currentValue < getSplitValue()) {
              (*leftPartitionIndex).push_back(*it);
            } else {
              (*rightPartitionIndex).push_back(*it);
            }
          }
        }
      }

    }

      // If we need to return the weightmatrix, do the same thing for the training data
      std::vector<size_t>* leftPartitionAveragingIndex = nullptr;
      std::vector<size_t>* rightPartitionAveragingIndex = nullptr;
      if (weightMatrix) {

          leftPartitionAveragingIndex = new std::vector<size_t>();
          rightPartitionAveragingIndex = new std::vector<size_t>();

          // Test if the splitting feature is categorical
          if (
                  categorical_split
                  ){

              for (
                      std::vector<size_t>::iterator it = (*predictionAveragingIndices).begin();
                      it != (*predictionAveragingIndices).end();
                      ++it
                      ) {

                  double currentValue = trainingData->getPoint(*it, getSplitFeature());

                  if (std::isnan(currentValue)){
                      size_t draw;

                      // If we have a missing feature value, if no NAs were observed when
                      // splitting send to left/right with probability proportional to
                      // number of observations in left/right child node else send
                      // right/left with probability in proportion to NA's which went
                      // left/right when splitting

                      if ((naLeftCount == 0) && (naRightCount == 0)) {
                          draw = discrete_dist_nonmissing(random_number_generator);
                      } else {
                          draw = discrete_dist(random_number_generator);
                      }

                      // Have to push to three indices when we have trinary splits
                      if (draw == 0) {
                          (*leftPartitionAveragingIndex).push_back(*it);
                      } else {
                          (*rightPartitionAveragingIndex).push_back(*it);
                      }

                  } else if (currentValue == getSplitValue()) {
                      (*leftPartitionAveragingIndex).push_back(*it);
                  } else {
                      (*rightPartitionAveragingIndex).push_back(*it);
                  }
              }

          } else {

              // For non-categorical, split to left (<) and right (>=) according to the
              // split value
              for (
                      std::vector<size_t>::iterator it = (*predictionAveragingIndices).begin();
                      it != (*predictionAveragingIndices).end();
                      ++it
                      ) {

                  double currentValue = trainingData->getPoint(*it, getSplitFeature());

                  if (std::isnan(currentValue)){
                      size_t draw;

                      // If we have a missing feature value, if no NAs were observed when
                      // splitting send to left/right with probability proportional to
                      // number of observations in left/right child node else send
                      // right/left with probability in proportion to NA's which went
                      // left/right when splitting
                      if (getTrinary()) {
                          if ((naLeftCount == 0) && (naRightCount == 0)) {
                              draw = discrete_dist_nonmissing(random_number_generator);
                          } else {
                              draw = discrete_dist(random_number_generator);
                          }

                          // Have to push to three indices when we have trinary splits
                          if (draw == -1) {
                              (*leftPartitionAveragingIndex).push_back(*it);
                          } else if (draw == 1) {
                              (*rightPartitionAveragingIndex).push_back(*it);
                          }
                      } else {
                          if ((naLeftCount == 0) && (naRightCount == 0)) {
                              draw = discrete_dist_nonmissing(random_number_generator);
                          } else {
                              draw = discrete_dist(random_number_generator);
                          }
                          // Now push to index
                          if (draw == 0) {
                              (*leftPartitionAveragingIndex).push_back(*it);
                          } else {
                              (*rightPartitionAveragingIndex).push_back(*it);
                          }
                      }

                  } else {
                      // Check if the current split feature is a symmetric feature
                      if (getTrinary() && (std::find(
                              trainingData->getSymmetricIndices()->begin(),
                              trainingData->getSymmetricIndices()->end(),
                              getSplitFeature()) != trainingData->getSymmetricIndices()->end())) {

                          // If this is a symmetric feature, have to use the absolute value
                          // of the feature value
                          if (std::fabs(currentValue) < getSplitValue()) {
                              (*leftPartitionAveragingIndex).push_back(*it);
                          } else {
                              (*rightPartitionAveragingIndex).push_back(*it);
                          }
                      } else {
                          // Run standard predictions
                          if (currentValue < getSplitValue()) {
                              (*leftPartitionAveragingIndex).push_back(*it);
                          } else {
                              (*rightPartitionAveragingIndex).push_back(*it);
                          }
                      }
                  }
              }

          }

      }

      // Recursively get predictions from its children
    if ((*leftPartitionIndex).size() > 0) {
      (*getLeftChild()).predict(
          outputPrediction,
          terminalNodes,
          outputCoefficients,
          leftPartitionIndex,
          leftPartitionAveragingIndex,
          xNew,
          trainingData,
          weightMatrix,
          linear,
          naDirection,
          lambda,
          seed,
          nodesizeStrictAvg,
          OOBIndex
      );
    }

    if ((*rightPartitionIndex).size() > 0) {
        (*getRightChild()).predict(
          outputPrediction,
          terminalNodes,
          outputCoefficients,
          rightPartitionIndex,
          rightPartitionAveragingIndex,
          xNew,
          trainingData,
          weightMatrix,
          linear,
          naDirection,
          lambda,
          seed,
          nodesizeStrictAvg,
          OOBIndex
        );
    }

    delete(leftPartitionIndex);
    delete(rightPartitionIndex);
    if (weightMatrix) {
        delete(leftPartitionAveragingIndex);
        delete(rightPartitionAveragingIndex);
    }

  }
}

bool RFNode::is_leaf() {
  int ave_ct = getAverageCount();
  //int spl_ct = getSplitCount();
  // if (
  //     (ave_ct == 0 && spl_ct != 0) ||(ave_ct != 0 && spl_ct == 0)
  // ) {
  //   throw std::runtime_error(
  //       "Average count or Split count is 0, while the other is not!"
  //       );
  // }
  //return !(ave_ct == 0 && spl_ct == 0);
  return !(ave_ct == 0);
}

size_t RFNode::getAverageCountAlways() {
  if(is_leaf()) {
    return _averageCount;
  }
  else {
    return (*getRightChild()).getAverageCountAlways() +
      (*getLeftChild()).getAverageCountAlways();
  }
}

void RFNode::printSubtree(int indentSpace) {

  // Test if the node is leaf node
  if (is_leaf()) {

    // Print count of samples in the leaf node
    RcppThread::Rcout << std::string((unsigned long) indentSpace, ' ')
              << "Leaf Node: # of split samples = "
              << getSplitCount()
              << ", # of average samples = "
              << getAverageCount()
              << " Weight = "
              << getPredictWeight()
              << std::endl;
    R_FlushConsole();
    R_ProcessEvents();

  } else {

    // Print split feature and split value
    RcppThread::Rcout << std::string((unsigned long) indentSpace, ' ')
              << "Tree Node: split feature = "
              << getSplitFeature()
              << ", split value = "
              << getSplitValue()
              << ", trinary indicator "
              << getTrinary()
              << ", # of average samples = "
              << getAverageCount()
              << ", # NA's l,r = "
              << getNaLeftCount()
              << " "
              << getNaRightCount()
              << " Weight = "
              << getPredictWeight()
              << std::endl;

    R_FlushConsole();
    R_ProcessEvents();
    // Recursively calling its children
    (*getLeftChild()).printSubtree(indentSpace+2);
    (*getRightChild()).printSubtree(indentSpace+2);
  }
}

// -----------------------------------------------------------------------------

void RFNode::write_node_info(
    std::unique_ptr<tree_info> & treeInfo,
    DataFrame* trainingData
){
  if (is_leaf()) {
    // If it is a leaf: set everything to be 0
    treeInfo->var_id.push_back(-getAverageCount());
    treeInfo->var_id.push_back(-getSplitCount());
    treeInfo->split_val.push_back(0);
    treeInfo->naLeftCount.push_back(-1);
    treeInfo->naRightCount.push_back(-1);
    treeInfo->naDefaultDirection.push_back(0);

    treeInfo->num_avg_samples.push_back(getAverageCount());
    treeInfo->num_spl_samples.push_back(getSplitCount());
    treeInfo->values.push_back(getPredictWeight());
  } else {
    // If it is a usual node: remember split var and split value and recursively
    // call write_node_info on the left and the right child.
    treeInfo->var_id.push_back(getSplitFeature() + 1);
    treeInfo->split_val.push_back(getSplitValue());
    treeInfo->naLeftCount.push_back(getNaLeftCount());
    treeInfo->naRightCount.push_back(getNaRightCount());
    treeInfo->naDefaultDirection.push_back(getNaDefaultDirection());

    getLeftChild()->write_node_info(treeInfo, trainingData);
    getRightChild()->write_node_info(treeInfo, trainingData);
  }
  return;
}
