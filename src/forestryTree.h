#ifndef HTECPP_RFTREE_H
#define HTECPP_RFTREE_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>

class forestryTree {

public:
  forestryTree();
  virtual ~forestryTree();

  forestryTree(
    DataFrame* trainingData,
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    double minSplitGain,
    size_t maxDepth,
    size_t interactionDepth,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool hasNas,
    bool naDirection,
    bool linear,
    bool symmetric,
    double overfitPenalty,
    unsigned int seed
  );

  // This tree is only for testing purpose
  void setDummyTree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    double minSplitGain,
    size_t maxDepth,
    size_t interactionDepth,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    double overfitPenalty
  );

  void predict(
    std::vector<double> &outputPrediction,
    std::vector<int>* terminalNodes,
    std::vector< std::vector<double> > &outputCoefficients,
    std::vector< std::vector<double> >* xNew,
    DataFrame* trainingData,
    arma::Mat<double>* weightMatrix = NULL,
    bool linear = false,
    bool naDirection = false,
    unsigned int seed = 44,
    size_t nodesizeStrictAvg = 1,
    std::vector<size_t>* OOBIndex = NULL
  );

  std::unique_ptr<tree_info> getTreeInfo(
      DataFrame* trainingData
  );

  void reconstruct_tree(
      size_t mtry,
      size_t minNodeSizeSpt,
      size_t minNodeSizeAvg,
      size_t minNodeSizeToSplitSpt,
      size_t minNodeSizeToSplitAvg,
      double minSplitGain,
      size_t maxDepth,
      size_t interactionDepth,
      bool hasNas,
      bool naDirection,
      bool linear,
      double overfitPenalty,
      unsigned int seed,
      std::vector<size_t> categoricalFeatureColsRcpp,
      std::vector<int> var_ids,
      std::vector<double> split_vals,
      std::vector<int> naLeftCounts,
      std::vector<int> naRightCounts,
      std::vector<int> naDefaultDirections,
      std::vector<size_t> averagingSampleIndex,
      std::vector<size_t> splittingSampleIndex,
      std::vector<double> predictWeights);

  void recursive_reconstruction(
      RFNode* currentNode,
      std::vector<int> * var_ids,
      std::vector<double> * split_vals,
      std::vector<int> * naLeftCounts,
      std::vector<int> * naRightCounts,
      std::vector<int> * naDefaultDirections,
      std::vector<double> * weights
  );

  void recursivePartition(
    RFNode* rootNode,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    size_t depth,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    double overfitPenalty,
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal,
    bool monotone_splits,
    monotonic_info monotone_details,
    bool trinary,
    bool centerSplit,
    symmetric_info symmetric_details,
    bool naDirection
  );

  void selectBestFeature(
      size_t &bestSplitFeature,
      double &bestSplitValue,
      double &bestSplitLoss,
      int &bestSplitNaDir,
      std::vector<double> &bestSplitLeftWts,
      std::vector<double> &bestSplitRightWts,
      arma::Mat<double> &bestSplitGL,
      arma::Mat<double> &bestSplitGR,
      arma::Mat<double> &bestSplitSL,
      arma::Mat<double> &bestSplitSR,
      std::vector<size_t>* featureList,
      std::vector<size_t>* averagingSampleIndex,
      std::vector<size_t>* splittingSampleIndex,
      DataFrame* trainingData,
      std::mt19937_64& random_number_generator,
      bool splitMiddle,
      size_t maxObs,
      bool linear,
      bool trinary,
      double overfitPenalty,
      std::shared_ptr< arma::Mat<double> > gtotal,
      std::shared_ptr< arma::Mat<double> > stotal,
      bool monotone_splits,
      monotonic_info &monotone_details,
      symmetric_info &symmetric_details
  );

  void initializelinear(
      DataFrame* trainingData,
      arma::Mat<double>& gTotal,
      arma::Mat<double>& sTotal,
      size_t numLinearFeatures,
      std::vector<size_t>* splitIndexes
  );

  void printTree();

  void trainTiming();

  void getOOBindex(
    std::vector<size_t> &outputOOBIndex,
    std::vector<size_t> &allIndex
  );

  void getDoubleOOBIndex(
      std::vector<size_t> &outputOOBIndex,
      std::vector<size_t> &allIndex
  );

  void getOOBhonestIndex(
      std::vector<size_t> &outputOOBIndex,
      std::vector<size_t> &allIndex
  );

  void getOOGIndex(
      std::vector<size_t> &outputOOBIndex,
      std::vector<size_t> groupMemberships,
      std::vector<size_t> &allIndex,
      bool doubleOOB
  );

  void getOOBPrediction(
    std::vector<double> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData,
    bool OOBhonest,
    bool doubleOOB,
    size_t nodesizeStrictAvg,
    std::vector< std::vector<double> >* xNew,
    arma::Mat<double>* weightMatrix,
    const std::vector<size_t>& training_idx
  );

  void getShuffledOOBPrediction(
      std::vector<double> &outputOOBPrediction,
      std::vector<size_t> &outputOOBCount,
      DataFrame* trainingData,
      size_t shuffleFeature,
      std::mt19937_64& random_number_generator,
      size_t nodesizeStrictAvg
  );

  size_t getMtry() {
    return _mtry;
  }

  size_t getMinNodeSizeSpt() {
    return _minNodeSizeSpt;
  }

  size_t getMinNodeSizeAvg() {
    return _minNodeSizeAvg;
  }

  size_t getMinNodeSizeToSplitSpt() {
    return _minNodeSizeToSplitSpt;
  }

  size_t getMinNodeSizeToSplitAvg() {
    return _minNodeSizeToSplitAvg;
  }

  double getMinSplitGain() {
    return _minSplitGain;
  }

  size_t getMaxDepth() {
    return _maxDepth;
  }

  size_t getInteractionDepth() {
    return _interactionDepth;
  }

  std::vector<size_t>* getSplittingIndex() {
    return _splittingSampleIndex.get();
  }

  std::vector<size_t>* getAveragingIndex() {
    return _averagingSampleIndex.get();
  }

  RFNode* getRoot() {
    return _root.get();
  }

  double getOverfitPenalty() {
    return _overfitPenalty;
  }

  unsigned int getSeed() {
    return _seed;
  }

  bool gethasNas() {
    return _hasNas;
  }

  bool getNaDirection() {
    return _naDirection;
  }

  void assignNodeId(size_t& node_i) {
    node_i = ++_nodeCount;
  }

  size_t getNodeCount() {
    return _nodeCount;
  }

private:
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  double _minSplitGain;
  size_t _maxDepth;
  size_t _interactionDepth;
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  std::unique_ptr< RFNode > _root;
  bool _hasNas;
  bool _naDirection;
  bool _linear;
  double _overfitPenalty;
  unsigned int _seed;
  size_t _nodeCount;
};


#endif //HTECPP_RFTREE_H
