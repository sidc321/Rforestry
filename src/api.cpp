#include <vector>
#include <string>
#include <new>
#include <iostream>
#include <random>
#include "forestry.h"
#include "DataFrame.h"
#include "utils.h"

extern "C"{

    void* get_data (
            double** arr,
            int n_rows,
            int n_cols
            ) {
        // Create Data: first n_cols - 1 are features, last is outcome
        std::vector<std::vector<double>> data_numpy;


        for (int j = 0; j<n_cols; j++) {
            std::vector<double> col;
            for (int i = 0; i<n_rows; i++){
                col.push_back(arr[i][j]);
            }
            data_numpy.push_back(col);
        }

        std::unique_ptr< std::vector< std::vector<double> > > featureData {
                new std::vector<std::vector<double> >
        };

        for (size_t i = 0; i < n_cols-1; i++) {
            featureData->push_back(data_numpy[i]);
        }

        size_t numRows = n_rows;
        size_t numColumns = n_cols-1;

        // Create outcome data
        std::unique_ptr< std::vector<double> > outcomeData {
                new std::vector<double>
        };

        for (size_t i = 0; i < numRows; i++) {
            outcomeData->push_back(data_numpy[n_cols-1][i]);
        }

        // Categorical features column
        std::unique_ptr< std::vector<size_t> > categoricalFeatureCols (
                new std::vector<size_t>
        );

        // seed RNG generator
        std::mt19937_64 random_number_generator;
        random_number_generator.seed(24750371);

        std::unique_ptr< std::vector<size_t> > linearFeatures (
                new std::vector<size_t>
        );

        std::unique_ptr< std::vector<double> > feature_weights (
                new std::vector<double>(numColumns, ((double) 1.0)/((double) numColumns))
        );

        std::unique_ptr< std::vector<size_t> > feature_weight_vars (
                new std::vector<size_t>
        );

        std::unique_ptr< std::vector<double> > obs_weights (
                new std::vector<double>(numRows, ((double) 1.0)/((double) numRows))
        );

        std::unique_ptr< std::vector<int> > monotone_constraints (
                new std::vector<int>(numColumns, 0)
        );

        std::unique_ptr< std::vector<size_t> > groups (
                new std::vector<size_t>(numRows, 0)
        );

        std::unique_ptr< std::vector<size_t> > symmetric_constraints (
                new std::vector<size_t>
        );

        DataFrame* test_df = new DataFrame(
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
        );
        return test_df;
    }


    void* train_forest(
                void* data_ptr,
                size_t ntree,
                bool replace,
                size_t sampSize,
                double splitRatio,
                bool OOBhonest,
                bool doubleBootstrap,
                size_t mtry,
                size_t minNodeSizeSpt,
                size_t minNodeSizeAvg,
                size_t minNodeSizeToSplitSpt,
                size_t minNodeSizeToSplitAvg,
                double minSplitGain,
                size_t maxDepth,
                size_t interactionDepth,
                unsigned int seed,
                size_t nthread,
                bool verbose,
                bool splitMiddle,
                size_t maxObs,
                size_t minTreesPerGroup,
                bool hasNas,
                bool linear,
                bool symmetric,
                double overfitPenalty,
                bool doubleTree
    ){
        DataFrame* test_df = reinterpret_cast<DataFrame* >(data_ptr);

        forestry* forest ( new (std::nothrow) forestry(
                test_df,
                ntree,
                replace,
                sampSize,
                splitRatio,
                OOBhonest,
                doubleBootstrap,
                mtry,
                minNodeSizeSpt,
                minNodeSizeAvg,
                minNodeSizeToSplitSpt,
                minNodeSizeToSplitAvg,
                minSplitGain,
                maxDepth,
                interactionDepth,
                seed,
                nthread,
                verbose,
                splitMiddle,
                maxObs,
                minTreesPerGroup,
                hasNas,
                linear,
                symmetric,
                overfitPenalty,
                doubleTree
        ));

        if (verbose)
            std::cout << forest << std::endl;

        return forest;
    }

    std::vector<double>* predict_forest(
            void* forest_pt,
            void* dataframe_pt,
            double** test_data,
            int num_test_rows,
            bool verbose
    ){
        if (verbose)
            std::cout << forest_pt << std::endl;

        forestry* forest = reinterpret_cast<forestry *>(forest_pt);
        DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_pt);

        forest->_trainingData = dta_frame;

        // Create Data
        std::vector<std::vector<double>> data_numpy;


        for (int j = 0; j<dta_frame->getNumColumns(); j++) {
            std::vector<double> col;
            for (int i = 0; i<num_test_rows; i++){
                col.push_back(test_data[i][j]);
            }
            data_numpy.push_back(col);
        }

        std::vector< std::vector<double> >* predi_data (
                new std::vector< std::vector<double> >
                );

        for (size_t i = 0; i < dta_frame->getNumColumns(); i++) {
            predi_data->push_back(data_numpy[i]);
        }


        std::vector<double>* testForestPrediction = forest->predict(
                predi_data,
                nullptr,
                nullptr,
                nullptr,
                1,
                0,
                true,
                false,
                nullptr
        );

        return testForestPrediction;
    }

}

