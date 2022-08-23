#include <vector>
#include <string>
#include <new>
#include <iostream>
#include <random>
#include "forestry.h"
#include "DataFrame.h"
#include "utils.h"
#include "forestryTree.h"


extern "C"{

void* get_data (
        double* arr,
        size_t* categorical_vars,
        size_t countCategoricals,
        size_t* linFeat_idx,
        size_t countLinFeats,
        double* feat_weights,
        size_t* feat_weight_vars,
        size_t countFtWeightVars,
        double* deep_feat_weights,
        size_t* deep_feat_weight_vars,
        size_t countDeepFtWeightVars,
        double* observation_weights,
        int* mon_constraints,
        size_t* groupMemberships,
        bool monotoneAvg,
        size_t* symmetricIndices,
        size_t countSym,
        size_t numRows,
        size_t numColumns,
        unsigned int seed
) {
    // Create Data: first n_cols - 1 are features, last is outcome
    std::unique_ptr< std::vector< std::vector<double> > > featureData {
            new std::vector< std::vector<double> >(numColumns-1, std::vector<double>(numRows))
    };

    for (size_t j = 0; j < numColumns-1; j++) {
        for (size_t i = 0; i<numRows; i++){
            featureData->at(j).at(i) = arr[i*numColumns + j];
        }
    }
    

    // Create outcome data
    std::unique_ptr< std::vector<double> > outcomeData {
            new std::vector<double>(numRows)
    };

    for (size_t i = 0; i < numRows; i++) {
        outcomeData->at(i) = arr[numColumns*i-1];
    }

    numColumns--;


    // Categorical features column
    std::unique_ptr< std::vector<size_t> > categoricalFeatureCols (
            new std::vector<size_t> (countCategoricals)
    );

    for (size_t i = 0; i < countCategoricals; i++) {
        categoricalFeatureCols->at(i) = categorical_vars[i];
    }


    // Linear features column
    std::unique_ptr< std::vector<size_t> > linearFeatures (
            new std::vector<size_t> (countLinFeats)
    );

    for (size_t i = 0; i < countLinFeats; i++) {
        linearFeatures->at(i) = linFeat_idx[i];
    }


    // Feature weights for each column
    std::unique_ptr< std::vector<double> > feature_weights (
            new std::vector<double>(numColumns)
    );

    for (size_t i = 0; i < numColumns; i++){
        feature_weights->at(i) = feat_weights[i];
    }


    // Feature indecies based on feature_weights
    std::unique_ptr< std::vector<size_t> > feature_weight_vars (
            new std::vector<size_t> (countFtWeightVars)
    );

    for (size_t i = 0; i < countFtWeightVars; i++) {
        feature_weight_vars->at(i) = feat_weight_vars[i];
    }


    // Deep feature weights for each column
    std::unique_ptr< std::vector<double> > deep_feature_weights (
            new std::vector<double>(numColumns)
    );

    for (size_t i = 0; i < numColumns; i++){
        deep_feature_weights->at(i) = deep_feat_weights[i];
    }


    // Deep feature indecies based
    std::unique_ptr< std::vector<size_t> > deep_feature_weight_vars (
            new std::vector<size_t> (countDeepFtWeightVars)
    );

    for (size_t i = 0; i < countDeepFtWeightVars; i++) {
        deep_feature_weight_vars->at(i) = deep_feat_weight_vars[i];
    }


    // Observation weights
    std::unique_ptr< std::vector<double> > obs_weights (
            new std::vector<double>(numRows)
    );

    for (size_t i = 0; i < numRows; i++){
        obs_weights->at(i) = observation_weights[i];
    }


    // monotone constraints for each column
    std::unique_ptr< std::vector<int> > monotone_constraints (
            new std::vector<int>(numColumns)
    );

    for (size_t i = 0; i < numColumns; i++){
        monotone_constraints->at(i) = mon_constraints[i];
    }


    // group membership for each observation
    std::unique_ptr< std::vector<size_t> > groups (
            new std::vector<size_t>(numRows)
    );

    for (size_t i = 0; i < numRows; i++){
        groups->at(i) = groupMemberships[i];
    }


    // symmetric variable indices
    std::unique_ptr< std::vector<size_t> > symmetric_constraints (
            new std::vector<size_t> (countSym)
    );

    for (size_t i = 0; i < countSym; i++) {
        symmetric_constraints->at(i) = symmetricIndices[i];
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
            std::move(deep_feature_weights),
            std::move(deep_feature_weight_vars),
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

    if (verbose) {
        std::cout << forest << std::endl;
        forest->getForest()->at(0)->printTree();
    }
    return forest;
}

void predict_forest(
        void* forest_pt,
        void* dataframe_pt,
        double* test_data,
        unsigned int seed,
        size_t nthread,
        bool exact,
        bool returnWeightMatrix,
        bool use_weights,
        size_t* tree_weights,
        size_t num_test_rows,
        double (&predictions)[],
        double (&weight_matrix)[]
){   


    forestry* forest = reinterpret_cast<forestry *>(forest_pt);
    DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_pt);

    forest->_trainingData = dta_frame;

    // Create Data
    size_t ncol = dta_frame->getNumColumns();
    std::vector< std::vector<double> >* predi_data {
            new std::vector< std::vector<double> >(ncol, std::vector<double>(num_test_rows))
    };

    for (size_t j = 0; j < ncol; j++) {
        for (size_t i = 0; i < num_test_rows; i++){
            predi_data->at(j).at(i) = test_data[i*ncol + j];
        }
    }


    // Initialize the weightMatrix, terminalNodes, coefficients
    arma::Mat<double> weightMatrix;
    arma::Mat<int> terminalNodes;
    arma::Mat<double> coefficients;

    // tree_weights vector
    std::vector<size_t>* weights (
            new std::vector<size_t>(forest->getNtree())
    );

    for (size_t i = 0; i < forest->getNtree(); i++){
        weights->at(i) = tree_weights[i];
    }

    
    if (returnWeightMatrix) {
        weightMatrix.zeros(num_test_rows, dta_frame->getNumRows());                
        forest->predict_forestry(
                predi_data,
                predictions,
                &weightMatrix,
                nullptr,
                nullptr,
                seed,
                nthread,
                exact,
                false,
                nullptr
        );

        size_t idx = 0;
        for (size_t i = 0; i < num_test_rows; i++){
            for (size_t j = 0; j < dta_frame->getNumRows(); j++){
                weight_matrix[idx] = weightMatrix(i,j);
                idx++;
            }
        }
        
    }

    else {
        forest->predict_forestry(
            predi_data,
            predictions,
            nullptr,
            nullptr,
            nullptr,
            seed,
            nthread,
            exact,
            use_weights,
            weights
        );

    }

    delete(predi_data);
    delete(weights);


    // predict_info* predictionResults = new predict_info;
    // predictionResults->predictions = testForestPrediction;
    // predictionResults->weightMatrix = &weightMatrix;
    // predictionResults->terminalNodes = &terminalNodes;
    // predictionResults->coefficients = &coefficients;

    // return (void*)predictionResults;
}


void predictOOB_forest(
        void* forest_pt,
        void* dataframe_pt,
        double* test_data,
        bool doubleOOB,
        bool exact,
        bool returnWeightMatrix,
        bool verbose,
        double (&predictions)[],
        double (&weight_matrix)[]
){
    if (verbose)
        std::cout << forest_pt << std::endl;

    forestry* forest = reinterpret_cast<forestry *>(forest_pt);
    DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_pt);
    forest->_trainingData = dta_frame;

    //Create Data
    size_t ncol = dta_frame->getNumColumns();
    std::vector< std::vector<double> >* predi_data {
            new std::vector< std::vector<double> >(ncol, std::vector<double>(dta_frame->getNumRows()))
    };

    for (size_t j = 0; j < ncol; j++) {
        for (size_t i = 0; i < dta_frame->getNumRows(); i++){
            predi_data->at(j).at(i) = test_data[i*ncol + j];
        }
    }

    // Initialize the weightMatrix
    arma::Mat<double> weightMatrix;

    if (returnWeightMatrix) {
        weightMatrix.zeros(dta_frame->getNumRows(), dta_frame->getNumRows());  
        
        forest->predictOOB_forestry(
                predi_data,
                predictions,
                &weightMatrix,
                doubleOOB,
                exact
        );

        size_t idx = 0;
        for (size_t i = 0; i < dta_frame->getNumRows(); i++){
            for (size_t j = 0; j < dta_frame->getNumRows(); j++){
                weight_matrix[idx] = weightMatrix(i,j);
                idx++;
            }
        }
    }

    else {
        forest->predictOOB_forestry(
            predi_data,
            predictions,
            nullptr,
            doubleOOB,
            exact
        );
    }

    delete(predi_data);

}

std::vector<double>* getVI(void* forest_pt){
    forestry* forest = reinterpret_cast<forestry *>(forest_pt);
    forest->calculateVariableImportance();

    std::vector<double> VI = forest->getVariableImportance();

    std::vector<double>* variableImportances(
            new std::vector<double>(VI)
    );

    return variableImportances;
}

int getTreeNodeCount(void* forest_ptr,
                     int tree_idx) {
    forestry* forest = reinterpret_cast<forestry *>(forest_ptr);

    return ((int) forest->getForest()->at(tree_idx)->getNodeCount());
}

std::vector<double>* get_tree_info(void* forest_ptr,
                                void* dataframe_ptr,
                                int tree_idx){
    forestry* forest = reinterpret_cast<forestry *>(forest_ptr);
    DataFrame* dta_frame = reinterpret_cast<DataFrame *>(dataframe_ptr);
    forest->_trainingData = dta_frame;

    std::unique_ptr<tree_info> info_holder;

    info_holder = forest->getForest()->at(tree_idx)->getTreeInfo(forest->getTrainingData());
    int num_nodes = forest->getForest()->at(tree_idx)->getNodeCount();
    
    std::vector<double>* treeInfo(
            new std::vector<double>(num_nodes*8)
    );

    for (int i = 0; i < num_nodes; i++) {
        treeInfo->at(i) = (double)info_holder->left_child_id.at(i);
        treeInfo->at(num_nodes+i) = (double)info_holder->right_child_id.at(i);
        treeInfo->at(num_nodes*2+i) = (double)info_holder->var_id.at(i);
        treeInfo->at(num_nodes*3+i) = (double)info_holder->num_avg_samples.at(i);
        treeInfo->at(num_nodes*4+i) = info_holder->split_val.at(i);
        treeInfo->at(num_nodes*5+i) = info_holder->values.at(i);
        treeInfo->at(num_nodes*6+i) = info_holder->naLeftCount.at(i);
        treeInfo->at(num_nodes*7+i) = info_holder->naLeftCount.at(i);
    }

    treeInfo->push_back((info_holder->splittingSampleIndex).size());
    for (size_t i = 0; i < (info_holder->splittingSampleIndex).size(); i++){
        treeInfo->push_back(info_holder->splittingSampleIndex.at(i));
    }

    treeInfo->push_back((info_holder->averagingSampleIndex).size());
    for (size_t i = 0; i < (info_holder->averagingSampleIndex).size(); i++){
        treeInfo->push_back(info_holder->averagingSampleIndex.at(i));
    }

    treeInfo->push_back(info_holder->seed);

    return treeInfo;
    

}

std::vector<size_t>* get_path(void* forest_ptr,
                           double* obs_ptr,
                           int tree_idx) {
    forestry* forest = reinterpret_cast<forestry *>(forest_ptr);

    std::vector<double>* observationDta = new std::vector<double>(forest->getTrainingData()->getNumColumns());

    for (size_t i = 0; i < forest->getTrainingData()->getNumColumns(); i++) {
        observationDta->at(i) = obs_ptr[i];
    }

    forestryTree* tree = (forest->getForest()->at(tree_idx)).get();
    
    std::vector<size_t> path;
    path.push_back(0);

    tree->getRoot()->getPath(path, observationDta, forest->getTrainingData(), forest->getSeed());    

    std::vector<size_t>* node_ids(
            new std::vector<size_t> (path)
    );

    return node_ids;

}

double get_prediction(void* prediction_ptr, int i){
    predict_info* predictionResults = reinterpret_cast<predict_info* >(prediction_ptr);
    return predictionResults->predictions->at(i);
}

double get_weightMatrix(void* prediction_ptr, size_t i, size_t j){
    predict_info* predictionResults = reinterpret_cast<predict_info* >(prediction_ptr);
    return predictionResults->weightMatrix->at(i, j);
}


void* py_reconstructree(void* data_ptr,
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
        bool doubleTree,
        size_t* tree_counts,
        double* thresholds,
        int* features,
        int* na_left_count,
        int* na_right_count,
        size_t* split_idx,
        size_t* average_idx,
        double* predict_weights,
        unsigned int* tree_seeds){

    // Do stuff
    DataFrame* df = reinterpret_cast<DataFrame* >(data_ptr);
    forestry* forest ( new (std::nothrow) forestry(
            df,
            0,
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

    std::vector<size_t>* categoricalColumns = df->getCatCols();

    std::unique_ptr< std::vector<size_t> > categoricalFeatureCols_copy(
      new std::vector<size_t>
    );
    for (size_t i = 0; i < categoricalColumns->size(); i++){
        categoricalFeatureCols_copy->push_back(categoricalColumns->at(i));
    }
    

    // Decode the forest data and create appropriate pointers
    std::unique_ptr< std::vector< std::vector<int> > > var_ids(
      new std::vector< std::vector<int> >
    );
    std::unique_ptr< std::vector< std::vector<double> > > split_vals(
        new std::vector< std::vector<double> >
    );
    std::unique_ptr< std::vector< std::vector<int> > > naLeftCounts(
        new std::vector< std::vector<int> >
    );
    std::unique_ptr< std::vector< std::vector<int> > > naRightCounts(
        new std::vector< std::vector<int> >
    );
    std::unique_ptr< std::vector< std::vector<size_t> > > averagingSampleIndex(
        new std::vector< std::vector<size_t> >
    );
    std::unique_ptr< std::vector< std::vector<size_t> > > splittingSampleIndex(
        new std::vector< std::vector<size_t> >
    );
    std::unique_ptr< std::vector<unsigned int> > treeSeeds(
        new std::vector<unsigned int>
    );
    std::unique_ptr< std::vector< std::vector<double> > > predictWeights(
        new std::vector< std::vector<double> >
    );

    // Reserve space for each of the vectors equal to ntree
    var_ids->reserve(ntree);
    split_vals->reserve(ntree);
    averagingSampleIndex->reserve(ntree);
    splittingSampleIndex->reserve(ntree);
    naLeftCounts->reserve(ntree);
    naRightCounts->reserve(ntree);
    treeSeeds->reserve(ntree);
    predictWeights->reserve(ntree);

    // Now actually populate the vectors
    size_t ind = 0, ind_s = 0, ind_a = 0;
    for(size_t i = 0; i < ntree; i++){
        std::vector<int> cur_var_ids(tree_counts[3*i], 0);
        std::vector<double> cur_split_vals(tree_counts[3*i], 0);
        std::vector<int> curNaLeftCounts(tree_counts[3*i], 0);
        std::vector<int> curNaRightCounts(tree_counts[3*i], 0);
        std::vector<size_t> curSplittingSampleIndex(tree_counts[3*i+1], 0);
        std::vector<size_t> curAveragingSampleIndex(tree_counts[3*i+2], 0);
        std::vector<double> cur_predict_weights(tree_counts[3*i], 0);

        for(size_t j = 0; j < tree_counts[3*i]; j++){
            cur_var_ids.at(j) = features[ind];
            cur_split_vals.at(j) = thresholds[ind];
            curNaLeftCounts.at(j) = na_left_count[ind];
            curNaRightCounts.at(j) = na_right_count[ind];
            cur_predict_weights.at(j) = predict_weights[ind];

            ind++;
        }

        for(size_t j = 0; j < tree_counts[3*i+1]; j++){
            curSplittingSampleIndex.at(j) = split_idx[ind_s];
            ind_s++;
        }

        for(size_t j = 0; j < tree_counts[3*i+2]; j++){
            curAveragingSampleIndex.at(j) = average_idx[ind_a];
            ind_a++;
        }

        var_ids->push_back(cur_var_ids);
        split_vals->push_back(cur_split_vals);
        naLeftCounts->push_back(curNaLeftCounts);
        naRightCounts->push_back(curNaRightCounts);
        splittingSampleIndex->push_back(curSplittingSampleIndex);
        averagingSampleIndex->push_back(curAveragingSampleIndex);
        predictWeights->push_back(cur_predict_weights);
        treeSeeds->push_back(tree_seeds[i]);
    }
    
    // call reconstructTrees
    forest->reconstructTrees(categoricalFeatureCols_copy,
                                   treeSeeds,
                                   var_ids,
                                   split_vals,
                                   naLeftCounts,
                                   naRightCounts,
                                   averagingSampleIndex,
                                   splittingSampleIndex,
                                   predictWeights
                                   );

    return forest;

}

int test_array_passing(double (&test_arr)[]){

    test_arr[1] = 1;

    return 0;
}

int test_array(size_t* arr){

    arr[0] = 1000;

    return 0;
}
    

}
