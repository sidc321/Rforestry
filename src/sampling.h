#ifndef SRC_SAMPLING_H
#define SRC_SAMPLING_H

#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <random>

void group_out_sample(
        size_t groupIdx,
        std::vector<size_t>& groupMemberships,
        std::vector<size_t>& outputIdx,
        std::mt19937_64& random_number_generator
);

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
);


#endif //SRC_SAMPLING_H
