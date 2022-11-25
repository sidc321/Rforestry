#include "sampling.h"
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

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
        std::mt19937_64& random_number_generator,
        DataFrame* trainingData
) {

    // Holds all of the indices that are out of the current fold
    std::vector<size_t> out_of_group_indices;

    // Gives the sampling weights for all indices that are out of the current fold
    std::vector<double> index_sampling_weights;

    // Get observation weights to use
    std::vector<double>* sampleWeights = (trainingData->getobservationWeights());

    // First get all observations not in groupIdx
    for (size_t i = 0; i < groupMemberships.size(); i++) {
        if (std::find(removedGroupIdx.begin(), removedGroupIdx.end(), groupMemberships[i]) == removedGroupIdx.end()) {
            out_of_group_indices.push_back(i);
            index_sampling_weights.push_back(sampleWeights->at(i));
        }
    }

    // Now sample the bootstrap sample from the out of group indices
    std::discrete_distribution<size_t> unif_dist(
            index_sampling_weights.begin(), index_sampling_weights.end()
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