#ifndef SRC_SAMPLING_H
#define SRC_SAMPLING_H

#include "dataFrame.h"
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

void assign_groups_to_folds(
        size_t numGroups,
        size_t foldSize,
        std::vector< std::vector<size_t> >& foldMemberships,
        std::mt19937_64& random_number_generator
);

void group_out_sample(
        std::vector<size_t>& removedGroupIdx,
        std::vector<size_t>& groupMemberships,
        std::vector<size_t>& outputIdx,
        std::mt19937_64& random_number_generator,
        DataFrame* trainingData
);

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
);

#endif //SRC_SAMPLING_H
