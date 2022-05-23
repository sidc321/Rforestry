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
            int* categorical_vars,
            int count_categorical_vars,
            int* linFeat_idx,
            int count_linFeat,
            double* feat_weights,
            int* feat_weight_vars,
            int count_feat_weight_vars,
            double* observation_weights,
            int* mon_constraints,
            int* groupMemberships,
            bool monotoneAvg,
            int* symmetricIndices,
            int countSymmetricIndices,
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

        size_t countCategoricals = count_categorical_vars;
        for (size_t i = 0; i < countCategoricals; i++) {
            categoricalFeatureCols->push_back(categorical_vars[i]);
        }


        // seed RNG generator
        std::mt19937_64 random_number_generator;
        random_number_generator.seed(24750371);

        // Linear features column
        std::unique_ptr< std::vector<size_t> > linearFeatures (
                new std::vector<size_t>
        );

        size_t countLinFeat = count_linFeat;
        for (size_t i = 0; i < countLinFeat; i++) {
            linearFeatures->push_back(linFeat_idx[i]);
        }

        // Feature weights for each column
        std::unique_ptr< std::vector<double> > feature_weights (
                new std::vector<double>(numColumns)
        );

        for (size_t i = 0; i < numColumns; i++)
        {
                feature_weights->at(i) = feat_weights[i];
        }
        

        // Feature indecies based on feature_weights
        std::unique_ptr< std::vector<size_t> > feature_weight_vars (
                new std::vector<size_t>
        );

        size_t countFtWeight = count_feat_weight_vars;
        for (size_t i = 0; i < countFtWeight; i++) {
            feature_weight_vars->push_back(feat_weight_vars[i]);
        }
        

        // Observation weights
        std::unique_ptr< std::vector<double> > obs_weights (
                new std::vector<double>(numRows)
        );

        for (size_t i = 0; i < numRows; i++)
        {
                obs_weights->at(i) = observation_weights[i];
        }
        

        // monotone constraints for each column
        std::unique_ptr< std::vector<int> > monotone_constraints (
                new std::vector<int>(numColumns)
        );

        for (size_t i = 0; i < numColumns; i++)
        {
                monotone_constraints->at(i) = mon_constraints[i];
        }


        // group membership for each observation
        std::unique_ptr< std::vector<size_t> > groups (
                new std::vector<size_t>(numRows)
        );

        for (size_t i = 0; i < numRows; i++)
        {
                groups->at(i) = groupMemberships[i];
        }


        // symmetric variable indices
        std::unique_ptr< std::vector<size_t> > symmetric_constraints (
                new std::vector<size_t>
        );

        size_t countSym = countSymmetricIndices;
        for (size_t i = 0; i < countSym; i++) {
            symmetric_constraints->push_back(symmetricIndices[i]);
        }


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
                monotoneAvg,
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

