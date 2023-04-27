#include "sampling.h"
#include "utils.h"
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_set>

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
// the removedGroupIdx group, this puts the resulting sample into outputIdx
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
    std::discrete_distribution<size_t> weighted_dist(
            index_sampling_weights.begin(), index_sampling_weights.end()
    );

    std::vector<size_t> sampledIndices;

    while (sampledIndices.size() < index_sampling_weights.size()) {
        size_t randomIndex = weighted_dist(random_number_generator);
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
        size_t numGroups,
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
    std::vector <size_t> sampleIndex;

    // If the forest is to be constructed with minTreesPerFold, we want to
    // use that sampling method instead of the sampling methods we have
    size_t currentFold;
    std::vector <size_t> groups_to_remove;

    // See if the weights are uniform or not
    std::unordered_set<double> uniqueWeights;
    std::vector<double>* sampleWeights = (trainingData->getobservationWeights());
    // Iterate through the vector and insert each value into the unordered_set
    for (const auto& value : (*sampleWeights)) {
        uniqueWeights.insert(value);
    }

    // use weights if multiple unique values
    bool use_weights = uniqueWeights.size() != 1;

    // If using groups with honesty, we split the groups into either splitting or
    // averaging groups before taking the bootstrap sample
    if (((*trainingData->getGroups())[0] != 0) && (oobHonest || ((splitratio != 1) && (splitratio != 0)))) {

        // If we are combining honesty with groups, first partition the groups of the tree into
        // splitting and averaging groups.
        if (oobHonest) {
            splitratio = .632;
        }

        // Now take the set of splitting and averaging groups based on the correct entry of honestGroupAssignments
        std::vector <size_t> splittingGroups;
        std::vector <size_t> averagingGroups;

        // Keep track of what vector has the splitting and averaging groups
        size_t splitFold = (splitratio >= 1 - splitratio ? 0 : 1);
        size_t avgFold = (splitratio >= 1 - splitratio ? 1 : 0);
        size_t honestSplitSize;

        // If we are doing leave fold-out-sampling, we split the groups into averaging and splitting sets
        // after removing the current fold of groups. Otherwise we just split the entire set of groups into averaging and splitting
        if ((minTreesPerFold > 0) && (treeIndex < groupToGrow)) {
            currentFold = (size_t) std::floor((double) treeIndex / (double) minTreesPerFold);
            std::vector <size_t> currentFoldGroups = foldMemberships[currentFold];

            honestSplitSize = (size_t) std::floor(
                    (std::max(splitratio, 1 - splitratio) * (double) (numGroups - currentFoldGroups.size())));

            // Holds the assignment of groups to either splitting or avging sets
            std::vector <std::vector<size_t>> honestGroupAssignments(2);
            honestGroupAssignments[0] = std::vector<size_t>(honestSplitSize);
            honestGroupAssignments[1] = std::vector<size_t>(honestSplitSize);

            // Now assign the groups to folds based on the smaller number of groups
            assign_groups_to_folds(numGroups - currentFoldGroups.size(),
                                   honestSplitSize,
                                   honestGroupAssignments,
                                   random_number_generator);

            // Now we need to actually fill in the correct groups, since we have sampled indices
            std::sort(currentFoldGroups.begin(), currentFoldGroups.end());

            std::vector <size_t> restrictedGroupIndices(numGroups);
            std::iota(restrictedGroupIndices.begin(), restrictedGroupIndices.end(), 1);

            // Get setDiff(1:numGroups, groupIndices of current fold)
            std::vector <size_t> restrictedGroupIndicesDiff;
            std::set_difference(restrictedGroupIndices.begin(),
                                restrictedGroupIndices.end(),
                                currentFoldGroups.begin(),
                                currentFoldGroups.end(),
                                std::inserter(restrictedGroupIndicesDiff, restrictedGroupIndicesDiff.begin()));

            // Go through folds and replace the indices we have sampled with the group they correspond to
            for (size_t i = 0; i < honestGroupAssignments[0].size(); i++) {
                honestGroupAssignments[0][i] = restrictedGroupIndicesDiff[honestGroupAssignments[0][i] - 1];
            }
            for (size_t i = 0; i < honestGroupAssignments[1].size(); i++) {
                honestGroupAssignments[1][i] = restrictedGroupIndicesDiff[honestGroupAssignments[1][i] - 1];
            }
            splittingGroups = honestGroupAssignments[splitFold];
            averagingGroups = honestGroupAssignments[avgFold];

        } else {
            honestSplitSize = (size_t) std::round((std::max(splitratio, 1 - splitratio) * (double) numGroups));

            // Avoid case where one of the sets is empty
            if (honestSplitSize == numGroups) {
                honestSplitSize = numGroups-1;
            } else if (honestSplitSize == 0) {
                honestSplitSize = 1;
            }

            // Holds the assignment of groups to either splitting or avging sets
            std::vector <std::vector<size_t>> honestGroupAssignments(2);
            honestGroupAssignments[0] = std::vector<size_t>(honestSplitSize);
            honestGroupAssignments[1] = std::vector<size_t>(honestSplitSize);


            // Partition groups randomly into the honesty sets. First index will always be the larger of the splitting
            // and averaging sets
            assign_groups_to_folds(numGroups,
                                   honestSplitSize,
                                   honestGroupAssignments,
                                   random_number_generator);

            splittingGroups = honestGroupAssignments[splitFold];
            averagingGroups = honestGroupAssignments[avgFold];
        }

        std::vector <size_t> splitSampleIndex_;
        std::vector <size_t> averageSampleIndex_;

        // Leave out the groups in the current fold when sampling
        std::vector <size_t> splitting_groups_to_remove = averagingGroups;

        // When sampling splitting indices, remove averaging groups
        if ((minTreesPerFold > 0) && (treeIndex < groupToGrow)) {
            currentFold = (size_t) std::floor((double) treeIndex / (double) minTreesPerFold);
            std::move((foldMemberships[currentFold]).begin(), (foldMemberships[currentFold]).end(),
                      std::back_inserter(splitting_groups_to_remove));
        }


        // Populate splitSampleIndex_ with the group_out_sample function
        // Here we hold out the averaging groups and the current leave out group
        group_out_sample(
                splitting_groups_to_remove,
                (*trainingData->getGroups()),
                splitSampleIndex_,
                random_number_generator,
                trainingData
        );

        std::vector <size_t> averaging_groups_to_remove = splittingGroups;
        // When sampling averaging indices, remove splitting groups
        if ((minTreesPerFold > 0) && (treeIndex < groupToGrow)) {
            currentFold = (size_t) std::floor((double) treeIndex / (double) minTreesPerFold);
            std::move((foldMemberships[currentFold]).begin(), (foldMemberships[currentFold]).end(),
                      std::back_inserter(averaging_groups_to_remove));
        }

        // Populate averageSampleIndex_ with the group_out_sample function
        // Here we hold out the splitting groups and the current leave out group
        group_out_sample(
                averaging_groups_to_remove,
                (*trainingData->getGroups()),
                averageSampleIndex_,
                random_number_generator,
                trainingData
        );

        // Set the indices and return, no need to get to the second part of the function where
        // we split into splitting and averaging
        splitSampleIndexReturn = splitSampleIndex_;
        averageSampleIndexReturn = averageSampleIndex_;
        return;

        // If sampling with groups or folds
    } else if ((minTreesPerFold > 0) && (treeIndex < groupToGrow)) {

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

        // If sampling is done with replacement and splitratio honesty, split the observations into
        // splitting and averaging and then bootstrap sample from each partition
    } else if (replacement && ((splitratio != 1) && (splitratio != 0))) {

        // Same machinery as partitioning groups, but at observation level
        std::vector <size_t> possibleSplittingIndices;
        std::vector <size_t> possibleAveragingIndices;

        // Keep track of what vector has the splitting and averaging indices
        size_t splitSetIdx = (splitratio >= 1 - splitratio ? 0 : 1);
        size_t avgSetIdx = (splitratio >= 1 - splitratio ? 1 : 0);
        size_t honestSplitSize = (size_t) std::floor((std::max(splitratio, 1 - splitratio) * (double) trainingData->getNumRows()));

        if (honestSplitSize == 0) {
            honestSplitSize = 1;
        }

        // Holds the assignment of indices to either splitting or avging sets
        std::vector <std::vector<size_t>> honestIndexAssignments(2);
        honestIndexAssignments[0] = std::vector<size_t>(honestSplitSize);
        honestIndexAssignments[1] = std::vector<size_t>(honestSplitSize);

        // Partition groups randomly into the honesty sets. First index will always be the larger of the splitting
        // and averaging sets
        assign_groups_to_folds(trainingData->getNumRows(),
                               honestSplitSize,
                               honestIndexAssignments,
                               random_number_generator);

        possibleSplittingIndices = honestIndexAssignments[splitSetIdx];
        possibleAveragingIndices = honestIndexAssignments[avgSetIdx];

        // Now carry out the sampling from the two partitions
        std::vector <size_t> splitSampleIndex_;
        std::vector <size_t> averageSampleIndex_;

        // Gives the sampling weights splitting + averaging partition indices
        std::vector<double> split_sampling_weights(possibleSplittingIndices.size());
        std::vector<double> avg_sampling_weights(possibleAveragingIndices.size());

        // Get observation weights to use
        std::vector<double>* sampleWeights = (trainingData->getobservationWeights());

        // Fill weights vectors with the correct observation weights
        for (size_t i = 0; i < possibleSplittingIndices.size(); i++) {
            split_sampling_weights[i] = sampleWeights->at(possibleSplittingIndices[i]-1);
        }
        for (size_t i = 0; i < possibleAveragingIndices.size(); i++) {
            avg_sampling_weights[i] = sampleWeights->at(possibleAveragingIndices[i]-1);
        }

        // Now sample the bootstrap sample from the two partitions
        std::discrete_distribution<size_t> weighted_split_dist(
                split_sampling_weights.begin(), split_sampling_weights.end()
        );
        std::discrete_distribution<size_t> weighted_avg_dist(
                avg_sampling_weights.begin(), avg_sampling_weights.end()
        );

        while (splitSampleIndex_.size() < split_sampling_weights.size()) {
            size_t randomIndex = weighted_split_dist(random_number_generator);
            // Push back the corresponding splitting Idx
            splitSampleIndex_.push_back(possibleSplittingIndices[randomIndex]-1);
        }
        while (averageSampleIndex_.size() < avg_sampling_weights.size()) {
            size_t randomIndex = weighted_avg_dist(random_number_generator);
            // Push back the corresponding averaging Idx
            averageSampleIndex_.push_back(possibleAveragingIndices[randomIndex]-1);
        }

        // Set the indices and return
        splitSampleIndexReturn = splitSampleIndex_;
        averageSampleIndexReturn = averageSampleIndex_;
        return;

    } else if (replacement && oobHonest && use_weights) {

        // Now we generate a weighted distribution using observationWeights
        std::vector<double>* sampleWeights = (trainingData->getobservationWeights());

        // First assign the unique observations into the splitting set, averaging set, or double OOB set for the tree
        // With the ratios:
        //  - Splitting set = .632
        //  - Averaging set = .233
        //  - Double OOB set = .135
        std::vector<size_t> all_unique_indices(sampleSize);

        // Create a random partition with the correct sizes by shuffling and taking the first
        // .135 for doob, next .233 for averaging, and next .632 for splitting
        std::iota(all_unique_indices.begin(), all_unique_indices.end(), 0);
        std::shuffle(all_unique_indices.begin(), all_unique_indices.end(), random_number_generator);

        size_t doob_count = std::max((size_t) 1, (size_t) std::floor(.135 * (double) sampleSize));
        size_t avg_count = std::max((size_t) 1, (size_t) std::floor(.233 * (double) sampleSize));
        size_t spl_count = sampleSize - doob_count - avg_count;

        // Create a vector of the unique avging + splitting indices for this tree
        std::vector<size_t> unique_avg_indices(all_unique_indices.begin() + doob_count,
                                               all_unique_indices.begin() + doob_count + avg_count);

        std::vector<size_t> unique_spl_indices(all_unique_indices.begin() + doob_count + avg_count,
                                               all_unique_indices.end());

        // Get the weights from the original weights vector and assign them to the unique averaging + splitting observations
        std::vector<double> potential_avg_weights(unique_avg_indices.size());
        for (size_t i = 0; i < unique_avg_indices.size(); i++) {
            potential_avg_weights[i] = sampleWeights->at(unique_avg_indices[i]);
        }
        std::vector<double> potential_spl_weights(unique_spl_indices.size());
        for (size_t i = 0; i < unique_spl_indices.size(); i++) {
            potential_spl_weights[i] = sampleWeights->at(unique_spl_indices[i]);
        }

        // Create weighted distribution over the potential averaging and splitting indices
        // Note it is okay not to explicitly normalize the weights since std::discrete_distribution does this already
        std::discrete_distribution<size_t> potential_avg_dist(
                potential_avg_weights.begin(), potential_avg_weights.end()
        );
        std::discrete_distribution<size_t> potential_spl_dist(
                potential_spl_weights.begin(), potential_spl_weights.end()
        );

        // Now carry out the sampling from the two partitions
        std::vector <size_t> splitSampleIndex_;
        std::vector <size_t> averageSampleIndex_;

        // Generate index with replacement for averaging set
        for (size_t j = 0; j < avg_count; j++) {
            size_t randomIndex = potential_avg_dist(random_number_generator);
            averageSampleIndex_.push_back(unique_avg_indices[randomIndex]);
        }

        // Generate index with replacement for averaging set
        for (size_t j = 0; j < spl_count; j++) {
            size_t randomIndex = potential_spl_dist(random_number_generator);
            splitSampleIndex_.push_back(unique_spl_indices[randomIndex]);
        }

        // Set the indices and return
        splitSampleIndexReturn = splitSampleIndex_;
        averageSampleIndexReturn = averageSampleIndex_;
        return;

        // Standard replacement sampling
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

            if (sampleIndex.size() == 0 ||
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

        // Treat it as normal RF
    } else if (splitratio == 1 || splitratio == 0) {
        splitSampleIndexReturn = sampleIndex;
        averageSampleIndexReturn = sampleIndex;
    }
}
