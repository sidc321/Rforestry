#ifndef FORESTRYCPP_RFNODE_H
#define FORESTRYCPP_RFNODE_H

#include <RcppArmadillo.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "DataFrame.h"
#include "utils.h"

class RFNode {

public:
  RFNode();
  virtual ~RFNode();

  void setLeafNode(
          size_t averagingSampleIndexSize,
          size_t splittingSampleIndexSize,
          size_t nodeId,
          bool trinary,
          std::vector<double> weights,
          double predictWeight
  );

  void setSplitNode(
      size_t splitFeature,
      double splitValue,
      std::unique_ptr< RFNode > leftChild,
      std::unique_ptr< RFNode > rightChild,
      bool trinary,
      size_t naLeftCount,
      size_t naCenterCount,
      size_t naRightCount
  );

  void setRidgeCoefficients(
          std::vector<size_t>* averagingIndices,
          DataFrame* trainingData,
          double lambda
  );


  void ridgePredict(
      std::vector<double> &outputPrediction,
      std::vector< std::vector<double> > &outputCoefficients,
      std::vector<size_t>* updateIndex,
      std::vector< std::vector<double> >* xNew,
      DataFrame* trainingData,
      double lambda
  );

  void predict(
    std::vector<double> &outputPrediction,
    std::vector<int>* terminalNodes,
    std::vector< std::vector<double> > &outputCoefficients,
    std::vector<size_t>* updateIndex,
    std::vector<size_t>* predictionAveragingIndices,
    std::vector< std::vector<double> >* xNew,
    DataFrame* trainingData,
    arma::Mat<double>* weightMatrix,
    bool linear,
    double lambda,
    unsigned int seed,
    size_t nodesizeStrictAvg,
    std::vector<size_t>* OOBIndex = NULL,
    bool fillRidgeCoefs = false
  );

  void write_node_info(
    std::unique_ptr<tree_info> & treeInfo,
    DataFrame* trainingData
  );

  bool is_leaf();

  void printSubtree(int indentSpace=0);

  size_t getSplitFeature() {
    if (is_leaf()) {
      throw "Cannot get split feature for a leaf.";
    } else {
      return _splitFeature;
    }
  }

  double getSplitValue() {
    if (is_leaf()) {
      throw "Cannot get split feature for a leaf.";
    } else {
      return _splitValue;
    }
  }

  RFNode* getLeftChild() {
    if (is_leaf()) {
      throw "Cannot get left child for a leaf.";
    } else {
      return _leftChild.get();
    }
  }

  RFNode* getRightChild() {
    if (is_leaf()) {
      throw "Cannot get right child for a leaf.";
    } else {
      return _rightChild.get();
    }
  }

  size_t getSplitCount() {
    return _splitCount;
  }

  size_t getAverageCount() {
    return _averageCount;
  }

  size_t getAverageCountAlways();

  size_t getNaLeftCount() {
    return _naLeftCount;
  }

  size_t getNaRightCount() {
    return _naRightCount;
  }

  size_t getNodeId() {
    return _nodeId;
  }

  std::vector<double> getWeights() {
    return _weights;
  }

  bool getTrinary() {
    return _trinary;
  }

  std::vector<size_t>* getAveragingIndex() {
    return _averagingSampleIndex.get();
  }

  std::vector<size_t>* getSplittingIndex() {
    return _splittingSampleIndex.get();
  }

  double getPredictWeight() {
      return _predictWeight;
  }

  arma::Mat<double> getRidgeCoefficients() {
      return _ridgeCoefficients;
  }

  arma::Mat<double> _ridgeCoefficients;

private:
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  size_t _splitFeature;
  double _splitValue;
  bool _trinary;
  double _predictWeight;
  std::vector<double> _weights;
  std::unique_ptr< RFNode > _leftChild;
  std::unique_ptr< RFNode > _rightChild;
  size_t _naLeftCount;
  size_t _naRightCount;
  size_t _averageCount;
  size_t _splitCount;
  size_t _nodeId;
};


#endif //FORESTRYCPP_RFNODE_H
