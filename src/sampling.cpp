#include "sampling.h"

#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>


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

void tree_sampling_helper(
        DataFrame* trainingData,
        size_t splitSampleSize,
        size_t minTreesPerGroup,
        size_t groupToGrow,
        size_t i,
        std::vector<size_t>& splitSampleIndex,
        std::vector<size_t>& averageSampleIndex,
        std::vector<size_t>& splitSampleIndex2,
        std::vector<size_t>& averageSampleIndex2,
        std::mt19937_64 random_number_generator,
        bool doubleTree,
        bool useReplacement,
        size_t sampleSize,
        double splitratio,
        bool OOBhonest,
        bool doubleBootstrap
) {
    // Generate a sample index for each tree
    std::vector<size_t> sampleIndex;
    // If the forest is to be constructed with minTreesPerGroup, we want to
    // use that sampling method instead of the sampling methods we have
    size_t currentGroup;

    if ((minTreesPerGroup > 0) && (i < groupToGrow)) {

        // Get the current group
        currentGroup = (((size_t) i) / ((size_t) minTreesPerGroup)) + 1;

        //RcppThread::Rcout << currentGroup;

        // Populate sampleIndex with the leave group out function
        group_out_sample(
                currentGroup,
                *(trainingData->getGroups()),
                sampleIndex,
                random_number_generator
        );

    } else if (useReplacement) {

        // Now we generate a weighted distribution using observationWeights
        std::vector<double>* sampleWeights = trainingData->getobservationWeights();
        std::discrete_distribution<size_t> sample_dist(
                sampleWeights->begin(), sampleWeights->end()
        );

        // Generate index with replacement
        while (sampleIndex.size() < sampleSize) {
            size_t randomIndex = sample_dist(random_number_generator);
            sampleIndex.push_back(randomIndex);
        }
    } else {
        // In this case, when we have no replacement, we disregard
        // observationWeights and use a uniform distribution
        std::uniform_int_distribution<size_t> unif_dist(
                0, (size_t) trainingData->getNumRows() - 1
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

    // If OOBhonest is true, we generate the averaging set based
    // on the OOB set.
    if (OOBhonest) {

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
            if (minTreesPerGroup == 0) {
                allIndex.push_back(i);
            } else if ((*(trainingData->getGroups()))[i]
                       != currentGroup) {
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
            // Now in new version, of OOB honesty
            // we want to sample with replacement from
            // the OOB index vector, so that our averaging vector
            // is also bagged.
            std::uniform_int_distribution<size_t> uniform_dist(
                    0, (size_t) (OOBIndex.size() - 1)
            );

            // Sample with replacement
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

        // Give split and avg sample indices the right indices
        splitSampleIndex = splitSampleIndex_;
        averageSampleIndex = averageSampleIndex_;

        // If we are doing doubleTree, swap the indices and make two trees
        if (doubleTree) {
            splitSampleIndex2 = splitSampleIndex_;
            averageSampleIndex2 = averageSampleIndex_;
        }
    } else if (splitratio == 1 || splitratio == 0) {

        // Treat it as normal RF
        splitSampleIndex = sampleIndex;
        averageSampleIndex = sampleIndex;

    } else {

        // Generate sample index based on the split ratio
        std::vector<size_t> splitSampleIndex_;
        std::vector<size_t> averageSampleIndex_;
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

        splitSampleIndex = splitSampleIndex_;
        averageSampleIndex = averageSampleIndex_;

        // If we are doing doubleTree, swap the indices and make two trees
        if (doubleTree) {
            splitSampleIndex2 = splitSampleIndex_;
            averageSampleIndex2 = averageSampleIndex_;
        }
    }

}
