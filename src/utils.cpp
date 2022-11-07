#include <Rcpp.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

void print_vector(
    std::vector<size_t> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
    // Rcpp's equivalent of std::flush
    R_FlushConsole();
    R_ProcessEvents();
    R_CheckUserInterrupt();
  }
  Rcpp::Rcout << std::endl;
  Rcpp::Rcout << std::endl;
}

void print_vector(
    std::vector<unsigned int> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
    // Rcpp's equivalent of std::flush
    R_FlushConsole();
    R_ProcessEvents();
    R_CheckUserInterrupt();
  }
  Rcpp::Rcout << std::endl;
  Rcpp::Rcout << std::endl;
}


void print_vector(
    std::vector<double> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
    // Rcpp's equivalent of std::flush
    R_FlushConsole();
    R_ProcessEvents();
    R_CheckUserInterrupt();
  }
  Rcpp::Rcout << std::endl;
  Rcpp::Rcout << std::endl;
}

int add_vector(
    std::vector<int>* v
) {
  int sum=0;
  for (size_t i = 0; i < v->size(); i++) {
    sum += (*v)[i];
  }
  return sum;
}

double square(
  double x
) {
  return (x*x);
}

// Given a number of groups, we assign each group to a
// fold of size foldSize (if numGroups % foldSize != 0, one
// fold may have a smaller number of groups).
// Returns a vector of vectors where the ith vector holds the
// indices of the groups in the ith fold.
void assign_groups_to_folds(
        size_t numGroups,
        size_t foldSize,
        std::vector< std::vector<size_t> >& foldMemberships,
        std::mt19937_64& random_number_generator
) {
    // Create vector of group indices
    std::vector<size_t> group_vector(numGroups);
    std::iota(group_vector.begin(), group_vector.end(), 1);

    // Shuffle the groups, now we can partition by foldSize to get a random partition
    std::shuffle(group_vector.begin(), group_vector.end(), random_number_generator);

    // Get number of folds we need
    size_t numFolds = (size_t) std::ceil((double) numGroups / (double) foldSize);

    // Populate foldMemberships with the partitioned shuffled values
    foldMemberships.reserve(numFolds);
    for (size_t iter = 0; iter < numFolds-1; iter++) {

        // Copy over the slice of the group vector to the appropriate vector in foldMemberships
        std::copy(group_vector.begin() + iter*foldSize,
                  group_vector.begin() + (iter+1)*foldSize,
                  foldMemberships[iter].begin());


    }

    // Last fold might be a bit smaller so copy based on group_vector.end()
    std::copy(group_vector.begin() + (numFolds-1)*foldSize,
              group_vector.end(),
              foldMemberships[numFolds-1].begin());
    foldMemberships[numFolds-1].resize(group_vector.size() - (numFolds-1)*foldSize);
}

// Does a bootstrap sample from the observations which do not fall into
// the groupIdx group, this puts the resulting sample into outputIdx
void group_out_sample(
    std::vector<size_t>& removedGroupIdx,
    std::vector<size_t>& groupMemberships,
    std::vector<size_t>& outputIdx,
    std::mt19937_64& random_number_generator
) {

  std::vector<size_t> out_of_group_indices;

  // First get all observations not in groupIdx
  for (size_t i = 0; i < groupMemberships.size(); i++) {
    if (std::find(removedGroupIdx.begin(), removedGroupIdx.end(), groupMemberships[i]) == removedGroupIdx.end()) {
      out_of_group_indices.push_back(i);
    }
  }

  // Now sample the bootstrap sample from the out of group indices
  std::uniform_int_distribution<size_t> unif_dist(
      0, (size_t) out_of_group_indices.size() - 1
  );

  std::vector<size_t> sampledIndices;

  while (sampledIndices.size() < groupMemberships.size()) {
    size_t randomIndex = unif_dist(random_number_generator);
    // Push back the out of group index at that position
      sampledIndices.push_back(out_of_group_indices[randomIndex]);
  }

  for (size_t i = 0; i < sampledIndices.size(); i++) {
    outputIdx.push_back(sampledIndices[i]);
  }
}

// helper function converting vector of binary flags to the size_t index
size_t bin_to_idx(
    std::vector<bool> binary
) {
  size_t idx = 0;
  size_t base = 1;
  for (const auto i : binary){
    if (i)
      idx += base;
    base *= 2;
  }
  return idx;
}

// Returns the binary entry at index i of the size_t idx
size_t idx_to_bin(
    size_t idx,
    size_t i
) {
  size_t bit = (size_t) pow(2, i); //(ith bit)
  if (i > 31) {
    return 0;
  } else {
    return (size_t (idx & bit) >= 1);
  }
}

// Given a feature vector and a list of symmetric features
// returns a vector of booleans indicating whether the symmetric feature
// at each index in symmmetric_indices is positive or negative
std::vector<bool> get_symmetric_feat_signs(
    std::vector<double> feature_vector,
    std::vector<size_t> symmmetric_indices
) {
  std::vector<bool> ret;
  for (const auto i : symmmetric_indices) {
    if (i >= feature_vector.size()) {
      return ret;
    } else {
      if (feature_vector[i] > 0) {
        ret.push_back(true);
      } else {
        ret.push_back(false);
      }
    }
  }
  return ret;
}

