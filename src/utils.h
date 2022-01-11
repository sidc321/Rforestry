#ifndef FORESTRYCPP_UTILS_H
#define FORESTRYCPP_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <iostream>

void print_vector(
  std::vector<size_t> v
);

void print_vector(
    std::vector<unsigned int> v
);

void print_vector(
    std::vector<double> v
);

int add_vector(
    std::vector<int>* v
);

double square(
    double x
);

void group_out_sample(
    size_t groupIdx,
    std::vector<size_t>& groupMemberships,
    std::vector<size_t>& outputIdx,
    std::mt19937_64& random_number_generator
);

size_t bin_to_idx(
    std::vector<bool> binary
);

size_t idx_to_bin(
    size_t idx,
    size_t i
);

std::vector<bool> get_symmetric_feat_signs(
    std::vector<double> feature_vector,
    std::vector<size_t> symmmetric_indices
);

struct tree_info {
  std::vector< int > var_id;
  // contains the variable id for a splitting node and the negative number of
  // observations in a leaf for a leaf node
  std::vector< long double > split_val;
  // contains the split values for regular nodes
  std::vector< int > leafAveidx;
  // contains the indices of observations in a leaf.
  std::vector< int > leafSplidx;
  // contains the indices of observations in a leaf.
  std::vector< int > averagingSampleIndex;
  // contains the indices of the average set.
  std::vector< int > splittingSampleIndex;
  // contains the indices of the splitting set.
  std::vector< int > naLeftCount;
  // Contains the count of NA's which fell to the left for each split value
  // (-1 indicates leaf node, 0 indicates no NA's fell that way)
  std::vector< int > naRightCount;
  // Contains the count of NA's which fell to the right for each split value
  // (-1 indicates leaf node, 0 indicates no NA's fell that way)
  unsigned int seed;
  // The seed that the tree was given (this uniquely identifies each tree
  // so that we can tell them apart. Very important for prediction when
  // exact = TRUE as we must aggregate the trees in the right order)
};

// Contains the information to help with monotonic constraints on splitting
struct monotonic_info {
  // Contains the monotonic constraints on each variable
  // For each continuous variable, we have +1 indicating a positive monotone
  // relationship, -1 indicating a negative monotone relationship, and 0
  // indicates no monotonic relationship
  std::vector<int> monotonic_constraints;

  // These contain the upper and lower bounds on node means for the node
  // currently being split on. These are used to reject potential splits
  // which do not respect the bounds, and therfore enforce global monotonic
  // bounds.
  double upper_bound;
  double lower_bound;

  // Two more upper bounds, these are only used when we are doing symmetric splits
  // and must enforce monotonicity on both the positive and negative outcomes
  // within a single node.
  double upper_bound_neg;
  double lower_bound_neg;
  // This flag indicates whether or not to enforce monotonicity on the averaging
  // set as well as the splitting set
  bool monotoneAvg;

  monotonic_info(){
    monotoneAvg = false;
  };
};

// Contains the information for symmetric splitting options
struct symmetric_info {

  // Contains the (0 indexed) indices of the variables that we are enforcing
  // symmetry on
  std::vector<size_t> symmetric_variables;

  std::vector<double> upper_bounds;
  std::vector<double> lower_bounds;

  // Contains the positive an negative pseudo outcomes for each combination of
  // signs of the symmetric features.
  // This is of size 2^|S| where S is the set of features with enforced symmetry
  // The integer expansion of the feature signs gives the index of the corresponding
  // pseudo outcome.
  // i.e. for two features, 0 = 00 is both features negative, 1 = 10 is feature
  // 1 positive feature 2 negative, 2 = 01 is feature 1 negative feature 2 positive,
  // and 3 = 11 is feature 1 and 2 positive.
  std::vector<double> pseudooutcomes;

};

#endif //FORESTRYCPP_UTILS_H
