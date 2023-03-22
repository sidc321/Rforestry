#include "utils.h"
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
    // std::cout << *i << ' ';
    // Rcpp's equivalent of std::flush
  }
  // std::cout << std::endl;
  // std::cout << std::endl;
}

void print_vector(
    std::vector<unsigned int> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    // std::cout << *i << ' ';
    // Rcpp's equivalent of std::flush
  }
  // std::cout << std::endl;
  // std::cout << std::endl;
}


void print_vector(
    std::vector<double> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    // std::cout << *i << ' ';
    // Rcpp's equivalent of std::flush
  }
  // std::cout << std::endl;
  // std::cout << std::endl;
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

