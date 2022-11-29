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

// Given a variety of hyperparameters, produce the indices of observations
// for a forestry tree to be grown using
void generate_sample_indices(
        std::vector<size_t>& splitSampleIndexReturn,
        std::vector<size_t>& averageSampleIndexReturn,
        size_t groupToGrow,
        size_t minTreesPerFold,
        size_t treeIndex,
        size_t sampleSize,
        bool replacement,
        bool oobHonest,
        bool doubleBootstrap,
        double splitratio,
        bool doubleTree,
        std::mt19937_64& random_number_generator,
        std::vector< std::vector<size_t> >& foldMemberships,
        DataFrame* trainingData
) {

    // Generate a sample index for each tree
    std::vector<size_t> sampleIndex;

    // If the forest is to be constructed with minTreesPerFold, we want to
    // use that sampling method instead of the sampling methods we have
    size_t currentFold;
    std::vector<size_t> groups_to_remove;

    size_t splitSampleSize = (size_t) (splitratio * sampleSize);

    // If sampling with groups or folds
    if ((minTreesPerFold > 0) && (treeIndex < groupToGrow)) {

        // Get the current fold
        currentFold = (size_t) std::floor((double) treeIndex / (double) minTreesPerFold);

        // Leave out the groups in the current fold when sampling
        groups_to_remove = (foldMemberships[currentFold]);

        // Populate sampleIndex with the group_out_sample function
        group_out_sample(
                groups_to_remove,
                (*trainingData->getGroups()),
                sampleIndex,
                random_number_generator,
                trainingData
        );

        // If sampling is done with replacement
    } else if (replacement) {

        // Now we generate a weighted distribution using observationWeights
        std::vector<double>* sampleWeights = (trainingData->getobservationWeights());
        std::discrete_distribution<size_t> sample_dist(
                sampleWeights->begin(), sampleWeights->end()
        );

        // Generate index with replacement
        while (sampleIndex.size() < sampleSize) {
            size_t randomIndex = sample_dist(random_number_generator);
            sampleIndex.push_back(randomIndex);
        }
        // If sampling is done without replacement
    } else {
        // When sampling without replacement, we disregard
        // observationWeights and use a uniform distribution
        std::uniform_int_distribution<size_t> unif_dist(
                0, (size_t) (*trainingData).getNumRows() - 1
        );

        // Generate index without replacement
        while (sampleIndex.size() < sampleSize) {
            size_t randomIndex = unif_dist(random_number_generator);

            if (
                    sampleIndex.size() == 0 ||
                    std::find(
                            sampleIndex.begin(),
                            sampleIndex.end(),
                            randomIndex
                    ) == sampleIndex.end()
                    ) {
                sampleIndex.push_back(randomIndex);
            }
        }
    }

    // Now split into splitting + averaging sets
    // If OOBhonest is true, we generate the averaging set based
    // on the OOB set.
    if (oobHonest) {

        std::vector<size_t> splitSampleIndex_;
        std::vector<size_t> averageSampleIndex_;

        std::sort(
                sampleIndex.begin(),
                sampleIndex.end()
        );

        std::vector<size_t> allIndex;
        for (size_t i = 0; i < sampleSize; i++) {
            // If we are doing leave a group out sampling, we make sure the
            // allIndex vector doesn't include observations in the currently
            // left out group
            if (minTreesPerFold == 0) {
                allIndex.push_back(i);
            } else if (std::find(groups_to_remove.begin(),
                                 groups_to_remove.end(),
                                 (*(trainingData->getGroups()))[i]) == groups_to_remove.end()) {
                allIndex.push_back(i);
            }
        }

        std::vector<size_t> OOBIndex(sampleSize);

        // First we get the set of all possible
        // OOB index is the set difference between sampleIndex and all_idx
        std::vector<size_t>::iterator it = std::set_difference (
                allIndex.begin(),
                allIndex.end(),
                sampleIndex.begin(),
                sampleIndex.end(),
                OOBIndex.begin()
        );

        // resize OOB index
        OOBIndex.resize((unsigned long) (it - OOBIndex.begin()));
        std::vector< size_t > AvgIndices;

        // Check the double bootstrap, if true, we take another sample
        // from the OOB indices, otherwise we just take the OOB index
        // set with standard (uniform) weightings
        if (doubleBootstrap) {
            std::uniform_int_distribution<size_t> uniform_dist(
                    0, (size_t) (OOBIndex.size() - 1)
            );

            // Sample with replacement from OOB Indices for the averaging set
            while (AvgIndices.size() < OOBIndex.size()) {
                size_t randomIndex = uniform_dist(random_number_generator);
                AvgIndices.push_back(
                        OOBIndex[randomIndex]
                );
            }
        } else {
            AvgIndices = OOBIndex;
        }

        // Now set the splitting indices and averaging indices
        splitSampleIndex_ = sampleIndex;
        averageSampleIndex_ = AvgIndices;

        splitSampleIndexReturn = splitSampleIndex_;
        averageSampleIndexReturn = averageSampleIndex_;

    } else if (splitratio == 1 || splitratio == 0) {

        // Treat it as normal RF
        splitSampleIndexReturn = sampleIndex;
        averageSampleIndexReturn = sampleIndex;

        // Standard Honesty - split the sampled indices into disjoint sets, splitting and averaging
    } else {

        // Generate sample index based on the split ratio
        std::vector<size_t> splitSampleIndex_;
        std::vector<size_t> averageSampleIndex_;

        // If we have groups, want to remove duplicates since sampleIndex
        // was sampled with replacement
        if (minTreesPerFold > 0 || replacement) {
            std::sort(sampleIndex.begin(), sampleIndex.end());
            sampleIndex.erase(std::unique(sampleIndex.begin(), sampleIndex.end()), sampleIndex.end());
            std::shuffle(sampleIndex.begin(), sampleIndex.end(), random_number_generator);
            splitSampleSize = (size_t) (splitratio * sampleIndex.size());
        }

        for (
                std::vector<size_t>::iterator it = sampleIndex.begin();
                it != sampleIndex.end();
                ++it
                ) {
            if (splitSampleIndex_.size() < splitSampleSize) {
                splitSampleIndex_.push_back(*it);
            } else {
                averageSampleIndex_.push_back(*it);
            }
        }
        splitSampleIndexReturn = splitSampleIndex_;
        averageSampleIndexReturn = averageSampleIndex_;

    }
}
