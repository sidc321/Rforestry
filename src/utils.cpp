#include <Rcpp.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>

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

// Does a bootstrap sample from the observations which do not fall into
// the groupIdx group, this puts the resulting sample into outputIdx
void group_out_sample(
    size_t groupIdx,
    std::vector<size_t>& groupMemberships,
    std::vector<size_t>& outputIdx,
    std::mt19937_64& random_number_generator
) {

  std::vector<size_t> out_of_group_indices;

  // First get all observations not in groupIdx
  for (size_t i = 0; i < groupMemberships.size(); i++) {
    if (groupMemberships[i] != groupIdx) {
      out_of_group_indices.push_back(i);
    }
  }

  // Now sample the bootstrap sample from the out of group indices
  std::uniform_int_distribution<size_t> unif_dist(
      0, (size_t) out_of_group_indices.size() - 1
  );

  std::vector<size_t> sampleIndex;

  while (sampleIndex.size() < groupMemberships.size()) {
    size_t randomIndex = unif_dist(random_number_generator);
    // Push back the out of group index at that position
    sampleIndex.push_back(out_of_group_indices[randomIndex]);
  }

  for (size_t i = 0; i < sampleIndex.size(); i++) {
    outputIdx.push_back(sampleIndex[i]);
  }
}

