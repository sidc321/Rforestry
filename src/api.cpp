#include <vector>
#include <string>
#include <iostream>
#include <random>
#include "forestry.h"
#include "DataFrame.h"
#include "utils.h"


extern "C" double train_forest(
        int i
){
    // Create Data
    size_t numRows = 150;
    size_t numColumns = 4;
    std::unique_ptr< std::vector< std::vector<double> > > featureData (
            new std::vector< std::vector<double> > {
                    {
                            5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                            5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
                            5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7,
                            4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1,
                            5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0,
                            7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
                            5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6,
                            5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7,
                            5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
                            5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7,
                            6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2,
                            6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0,
                            6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
                            7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9,
                            6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9
                    },
                    {
                            3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                            3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8,
                            3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2,
                            3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3.0, 3.4,
                            3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3,
                            3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7,
                            2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5,
                            3.2, 2.8, 2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6,
                            2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5,
                            2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8,
                            3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6,
                            3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2,
                            3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0,
                            2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1,
                            3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0
                    },
                    {
                            1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                            1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5,
                            1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6,
                            1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3, 1.5,
                            1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4,
                            4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9,
                            3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9,
                            4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5,
                            3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0,
                            4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1,
                            6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1,
                            5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0,
                            5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8,
                            6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4,
                            5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1
                    },
                    {
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3
                    }
            });

    // Create outcome data
    std::unique_ptr< std::vector<double> > outcomeData (
            new std::vector<double> {
                    0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                    0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3,
                    0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2,
                    0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2,
                    0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
                    1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4,
                    1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1,
                    1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0,
                    1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3,
                    1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3,
                    2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5,
                    2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5,
                    2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6,
                    1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1,
                    2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8
            });

    // Categorical features column
    std::unique_ptr< std::vector<size_t> > categoricalFeatureCols (
            new std::vector<size_t> {3}
    );

    // seed RNG generator
    std::mt19937_64 random_number_generator;
    random_number_generator.seed(24750371);

    std::unique_ptr< std::vector<size_t> > linearFeatures (
            new std::vector<size_t>(4, 0)
            );

    std::unique_ptr< std::vector<double> > feature_weights (
            new std::vector<double>(4, 0.25)
    );

    std::unique_ptr< std::vector<size_t> > feature_weight_vars (
            new std::vector<size_t>{0,1,2,3}
    );

    std::unique_ptr< std::vector<double> > obs_weights (
            new std::vector<double>(150, 0.006666667)
    );

    std::unique_ptr< std::vector<int> > monotone_constraints (
            new std::vector<int>(4, 0)
    );

    std::unique_ptr< std::vector<size_t> > groups (
            new std::vector<size_t>(150, 0)
    );

    std::unique_ptr< std::vector<size_t> > symmetric_constraints (
            new std::vector<size_t>(4, 0)
    );

    std::unique_ptr< DataFrame > test_df (new DataFrame(
            std::move(featureData),
            std::move(outcomeData),
            std::move(categoricalFeatureCols),
            std::move(linearFeatures),
            numRows,
            numColumns,
            std::move(feature_weights),
            std::move(feature_weight_vars),
            std::move(feature_weights),
            std::move(feature_weight_vars),
            std::move(obs_weights),
            std::move(monotone_constraints),
            std::move(groups),
            false,
            std::move(symmetric_constraints)
    ));

    std::unique_ptr<forestry> forest ( new forestry(
            test_df.get(),
            500,
            true,
            numRows,
            1,
            true,
            true,
            1,
            1,
            1,
            1,
            1,
            0.0,
            10,
            1,
            1,
            0,
            false,
            true,
            numRows,
            1,
            false,
            false,
            false,
            1.0,
            false
    ));

    std::unique_ptr< std::vector<double> > testForestPrediction (
            (*forest).predict(
                    (*forest).getTrainingData()->getAllFeatureData(),
                    nullptr,
                    nullptr,
                    nullptr,
                    1,
                    0,
                    false,
                    false,
                    nullptr
            )
    );

    // Calculate prediction accuracy
    std::vector<double>* truePredictions =
            (*forest).getTrainingData()->getOutcomeData();

    double testMSE = 0;
    for (size_t i=0; i<(*truePredictions).size(); i++){
        testMSE += pow(
                ((*testForestPrediction.get())[i] - (*truePredictions)[i]), 2
        );
    }

    return testMSE;
}