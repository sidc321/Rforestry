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
    std::vector<std::vector<double>> data_numpy;


    for (int j = 0; j<numColumns; j++) {
        std::vector<double> col;
        for (int i = 0; i<numRows; i++){
            col.push_back(arr[i*numColumns + j]);
        }
        data_numpy.push_back(col);
    }

    std::unique_ptr< std::vector< std::vector<double> > > featureData {
            new std::vector<std::vector<double> >
    };


    for (size_t i = 0; i < numColumns-1; i++) {
        featureData->push_back(data_numpy[i]);
    }

    numColumns--;

    // Create outcome data
    std::unique_ptr< std::vector<double> > outcomeData {
            new std::vector<double>(numRows)
    };

    for (size_t i = 0; i < numRows; i++) {
        outcomeData->at(i) = data_numpy[numColumns][i];
    }


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

    for (size_t i = 0; i < numColumns; i++)
    {
        feature_weights->at(i) = feat_weights[i];
    }


    // Feature indecies based on feature_weights
    std::unique_ptr< std::vector<size_t> > feature_weight_vars (
            new std::vector<size_t> (countFtWeightVars)
    );

    for (size_t i = 0; i < countFtWeightVars; i++) {
        feature_weight_vars->at(i) = feat_weight_vars[i];
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

    if (verbose) {
        std::cout << forest << std::endl;
        forest->getForest()->at(0)->printTree();
    }
    return forest;
}

std::vector<double>* predict_forest(
        void* forest_pt,
        void* dataframe_pt,
        double* test_data,
        unsigned int seed,
        size_t nthread,
        bool exact,
        bool use_weights,
        size_t* tree_weights,
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
            col.push_back(test_data[i*dta_frame->getNumColumns()+j]);
        }
        data_numpy.push_back(col);
    }

    std::vector< std::vector<double> >* predi_data (
            new std::vector< std::vector<double> >
    );

    for (size_t i = 0; i < dta_frame->getNumColumns(); i++) {
        predi_data->push_back(data_numpy[i]);
    }


    // tree_weights vector
    std::vector<size_t>* weights (
            new std::vector<size_t>(forest->getNtree())
    );

    for (size_t i = 0; i < forest->getNtree(); i++)
    {
        weights->at(i) = tree_weights[i];
    }


    std::vector<double>* testForestPrediction = forest->predict(
            predi_data,
            nullptr,
            nullptr,
            nullptr,
            seed,
            nthread,
            exact,
            use_weights,
            weights
    );

    return testForestPrediction;
}



std::vector<double>* predictOOB_forest(
        void* forest_pt,
        void* dataframe_pt,
        double* test_data,
        bool doubleOOB,
        bool exact,
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
            col.push_back(test_data[i*dta_frame->getNumColumns()+j]);
        }
        data_numpy.push_back(col);
    }

    std::vector< std::vector<double> >* predi_data (
            new std::vector< std::vector<double> >
    );

    for (size_t i = 0; i < dta_frame->getNumColumns(); i++) {
        predi_data->push_back(data_numpy[i]);
    }



    std::vector<double> testForestPredictionOOB = forest->predictOOB(
            predi_data,
            nullptr,
            doubleOOB,
            exact
    );

    std::vector<double>* preds (
            new std::vector<double>(num_test_rows)
    ) ;

    for (size_t i = 0; i < num_test_rows; i++){
        preds->at(i) = testForestPredictionOOB[i];
    }


    return preds;
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
    
    std::vector<double>* tree_info(
            new std::vector<double>(num_nodes*6)
    );

    for (int i = 0; i < num_nodes; i++) {
        tree_info->at(i) = (double)info_holder->left_child_id.at(i);
        tree_info->at(num_nodes+i) = (double)info_holder->right_child_id.at(i);
        tree_info->at(num_nodes*2+i) = (double)info_holder->var_id.at(i);
        tree_info->at(num_nodes*3+i) = (double)info_holder->num_avg_samples.at(i);
        tree_info->at(num_nodes*4+i) = info_holder->split_val.at(i);
        tree_info->at(num_nodes*5+i) = info_holder->values.at(i);
    }

    return tree_info;
    

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


}
