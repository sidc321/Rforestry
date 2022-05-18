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
            std::cout << data_numpy[n_cols-1][i] << std::endl;
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
            int input,
            void* data_ptr
    ){
        DataFrame* test_df = reinterpret_cast<DataFrame* >(data_ptr);
        int numRows = test_df->getNumRows();

        forestry* forest ( new (std::nothrow) forestry(
                test_df,
                input,
                true,
                numRows,
                1,
                false,
                false,
                1,
                1,
                1,
                1,
                1,
                0.0,
                10,
                100,
                1,
                0,
                true,
                true,
                numRows,
                0,
                false,
                false,
                false,
                1.0,
                false
        ));

        std::vector< std::unique_ptr< forestryTree > >* curr_forest;
        curr_forest = forest->getForest();
        std::sort(curr_forest->begin(), curr_forest->end(), [](const std::unique_ptr< forestryTree >& a,
                                                               const std::unique_ptr< forestryTree >& b) {
            return a.get()->getSeed() > b.get()->getSeed();
        });

        (*curr_forest)[0]->printTree();

        std::cout << forest << std::endl;
        return forest;
    }

    // std::vector<double>*

    std::vector<double>* predict_forest(
            void* forest_pt,
            void* dataframe_pt,
            double** test_data,
            int num_test_rows
    ){
        std::cout << forest_pt << std::endl;
        forestry* forest = reinterpret_cast<forestry *>(forest_pt);
        DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_pt);

        forest->_trainingData = dta_frame;

        //return forest->getMinNodeSizeSpt();

        size_t numRows = 150;
        std::cout << (*forest->getTrainingData()->getOutcomeData())[1] << std::endl;

        std::vector< std::unique_ptr< forestryTree > >* curr_forest;
        curr_forest = forest->getForest();
        (*curr_forest)[0]->printTree();

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


        //return (*predi_data)[1][1];

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


        std::vector<double>* ret_test(
                new std::vector<double>(numRows, 2.3)
        );

    /*    for (size_t i = 0; i <numRows; i++) {
            (*ret_test)[i] = (*testForestPrediction.get())[i];
            std::cout << (*testForestPrediction.get())[i] << std::endl;
        }*/

        return testForestPrediction;
    }

}

