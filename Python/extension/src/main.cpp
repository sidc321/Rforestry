#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "utils.cpp"
#include "api.cpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void* get_data_wrapper (
        py::array_t<double> arr,
        py::array_t<size_t> categorical_vars,
        size_t countCategoricals,
        py::array_t<size_t> linFeat_idx,
        size_t countLinFeats,
        py::array_t<double> feat_weights,
        py::array_t<size_t> feat_weight_vars,
        size_t countFtWeightVars,
        py::array_t<double> deep_feat_weights,
        py::array_t<size_t> deep_feat_weight_vars,
        size_t countDeepFtWeightVars,
        py::array_t<double> observation_weights,
        py::array_t<int> mon_constraints,
        py::array_t<size_t> groupMemberships,
        bool monotoneAvg,
        py::array_t<size_t> symmetricIndices,
        size_t countSym,
        size_t numRows,
        size_t numColumns,
        unsigned int seed
) {
    return get_data(
        static_cast<double *>(arr.request().ptr),
        static_cast<size_t *>(categorical_vars.request().ptr),
        countCategoricals,
        static_cast<size_t *>(linFeat_idx.request().ptr),
        countLinFeats,
        static_cast<double *>(feat_weights.request().ptr),
        static_cast<size_t *>(feat_weight_vars.request().ptr),
        countFtWeightVars,
        static_cast<double *>(deep_feat_weights.request().ptr),
        static_cast<size_t *>(deep_feat_weight_vars.request().ptr),
        countDeepFtWeightVars,
        static_cast<double *>(observation_weights.request().ptr),
        static_cast<int *>(mon_constraints.request().ptr),
        static_cast<size_t *>(groupMemberships.request().ptr),
        monotoneAvg,
        static_cast<size_t *>(symmetricIndices.request().ptr),
        countSym,
        numRows,
        numColumns,
        seed
    );
}

void* py_reconstructree_wrapper(void* data_ptr,
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
        py::array_t<size_t> tree_counts,
        py::array_t<double> thresholds,
        py::array_t<int> features,
        py::array_t<int> na_left_count,
        py::array_t<int> na_right_count,
        py::array_t<size_t> split_idx,
        py::array_t<size_t> average_idx,
        py::array_t<double> predict_weights,
        py::array_t<unsigned int> tree_seeds
) {
    return py_reconstructree(data_ptr,
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
        doubleTree,
        static_cast<size_t *>(tree_counts.request().ptr),
        static_cast<double *>(thresholds.request().ptr),
        static_cast<int *>(features.request().ptr),
        static_cast<int *>(na_left_count.request().ptr),
        static_cast<int *>(na_right_count.request().ptr),
        static_cast<size_t *>(split_idx.request().ptr),
        static_cast<size_t *>(average_idx.request().ptr),
        static_cast<double *>(predict_weights.request().ptr),
        static_cast<unsigned int *>(tree_seeds.request().ptr)
    );
}

void predictOOB_forest_wrapper(
        void* forest_pt,
        void* dataframe_pt,
        py::array_t<double> test_data,
        bool doubleOOB,
        bool exact,
        bool returnWeightMatrix,
        bool verbose,
        py::array_t<double> predictions,
        py::array_t<double> weight_matrix
) {
    py::buffer_info predictions_info = predictions.request();
    auto predictions_ptr = static_cast<double *>(predictions_info.ptr); 
    size_t predictions_size = predictions_info.shape[0];

    std::vector<double> predictions_copy(predictions_size);
    for (int i = 0; i < predictions_size; i++) {
        predictions_copy[i] = predictions_ptr[i];
    }

    py::buffer_info weight_matrix_info = weight_matrix.request();
    auto weight_matrix_ptr = static_cast<double *>(weight_matrix_info.ptr); 
    size_t weight_matrix_size = weight_matrix_info.shape[0];

    std::vector<double> weight_matrix_copy(weight_matrix_size);
    for (int i = 0; i < weight_matrix_size; i++) {
        weight_matrix_copy[i] = weight_matrix_ptr[i];
    }

    predictOOB_forest(
        forest_pt,
        dataframe_pt,
        static_cast<double *>(test_data.request().ptr),
        doubleOOB,
        exact,
        returnWeightMatrix,
        verbose,
        predictions_copy,
        weight_matrix_copy
    );

    for (int i = 0; i < predictions_size; i++) {
        predictions_ptr[i] = predictions_copy[i];
    }
    for (int i = 0; i < weight_matrix_size; i++) {
        weight_matrix_ptr[i] = weight_matrix_copy[i];
    }
}


void show_array(double *ptr, size_t size) {
    for(int i = 0; i < size; i++) {
        std::cout << *ptr << " ";
        ptr++;
    }
    std::cout << '\n' << '\n';
}

void show_array(std::vector<double> array) {
    for (auto i: array) {
        std::cout << i << ' ';
    }
    std::cout << '\n' << '\n';
}

void predict_forest_wrapper(
        void* forest_pt,
        void* dataframe_pt,
        py::array_t<double> test_data,
        unsigned int seed,
        size_t nthread,
        bool exact,
        bool returnWeightMatrix,
        bool linear,
        bool use_weights,
        py::array_t<size_t> tree_weights,
        size_t num_test_rows,
        py::array_t<double> predictions,
        py::array_t<double> weight_matrix,
        py::array_t<double> coefs
) {
    py::buffer_info predictions_info = predictions.request();
    double *predictions_ptr = static_cast<double *>(predictions_info.ptr); 
    size_t predictions_size = predictions_info.shape[0];

    std::vector<double> predictions_copy(predictions_size);
    for (int i = 0; i < predictions_size; i++) {
        predictions_copy[i] = predictions_ptr[i];
    }

    py::buffer_info weight_matrix_info = weight_matrix.request();
    auto weight_matrix_ptr = static_cast<double *>(weight_matrix_info.ptr); 
    size_t weight_matrix_size = weight_matrix_info.shape[0];

    std::vector<double> weight_matrix_copy(weight_matrix_size);
    for (int i = 0; i < weight_matrix_size; i++) {
        weight_matrix_copy[i] = weight_matrix_ptr[i];
    }

    py::buffer_info coefs_info = coefs.request();
    auto coefs_ptr = static_cast<double *>(coefs_info.ptr); 
    size_t coefs_size = coefs_info.shape[0];

    std::vector<double> coefs_copy(coefs_size);
    for (int i = 0; i < coefs_size; i++) {
        coefs_copy[i] = coefs_ptr[i];
    }

    predict_forest(
        forest_pt,
        dataframe_pt,
        static_cast<double *>(test_data.request().ptr),
        seed,
        nthread,
        exact,
        returnWeightMatrix,
        linear,
        use_weights,
        static_cast<size_t *>(tree_weights.request().ptr),
        num_test_rows,
        predictions_copy,
        weight_matrix_copy,
        coefs_copy
    );

    for (int i = 0; i < predictions_size; i++) {
        predictions_ptr[i] = predictions_copy[i];
    }

    for (int i = 0; i < weight_matrix_size; i++) {
        weight_matrix_ptr[i] = weight_matrix_copy[i];
    }
    for (int i = 0; i < coefs_size; i++) {
        coefs_ptr[i] = coefs_copy[i];
    }
}


void fill_tree_info_wrapper(void* forest_ptr,
                    int tree_idx,
                    py::array_t<double> treeInfo,
                    py::array_t<int> split_info,
                    py::array_t<int> av_info
) {
    py::buffer_info treeInfo_info = treeInfo.request();
    auto treeInfo_ptr = static_cast<double *>(treeInfo_info.ptr); 
    size_t treeInfo_size = treeInfo_info.shape[0];

    std::vector<double> treeInfo_copy(treeInfo_size);
    for (int i = 0; i < treeInfo_size; i++) {
        treeInfo_copy[i] = treeInfo_ptr[i];
    }

    py::buffer_info split_info_info = split_info.request();
    auto split_info_ptr = static_cast<int *>(split_info_info.ptr); 
    size_t split_info_size = split_info_info.shape[0];

    std::vector<int> split_info_copy(split_info_size);
    for (int i = 0; i < split_info_size; i++) {
        split_info_copy[i] = split_info_ptr[i];
    }

    py::buffer_info av_info_info = av_info.request();
    auto av_info_ptr = static_cast<int *>(av_info_info.ptr); 
    size_t av_info_size = av_info_info.shape[0];

    std::vector<int> av_info_copy(av_info_size);
    for (int i = 0; i < av_info_size; i++) {
        av_info_copy[i] = av_info_ptr[i];
    }

    fill_tree_info(forest_ptr,
                   tree_idx,
                   treeInfo_copy,
                   split_info_copy,
                   av_info_copy
    );

    for (int i = 0; i < treeInfo_size; i++) {
        treeInfo_ptr[i] = treeInfo_copy[i];
    }
    for (int i = 0; i < split_info_size; i++) {
        split_info_ptr[i] = split_info_copy[i];
    }
    for (int i = 0; i < av_info_size; i++) {
        av_info_ptr[i] = av_info_copy[i];
    }
}



PYBIND11_MODULE(extension, m) {
    m.doc() = R"pbdoc(
        RForestry Python extension module
        -----------------------

        .. currentmodule:: RForestry.extension

        .. autosummary::
           :toctree: _generate

           vector_get
    )pbdoc";

    m.def("train_forest", &train_forest, R"pbdoc(
        Some help text here
    
        Some other explanation about the train_forest function.
    )pbdoc");
    m.def("get_data", &get_data_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the get_data function.
    )pbdoc");
    m.def("vector_get", &vector_get, R"pbdoc(
        Some help text here

        Some other explanation about the get_vector function.
    )pbdoc");
    m.def("vector_get_int", &vector_get_int, R"pbdoc(
        Some help text here

        Some other explanation about the get_vector_int function.
    )pbdoc");
    m.def("vector_get_size_t", &vector_get_size_t, R"pbdoc(
        Some help text here

        Some other explanation about the get_vector_get_size_t function.
    )pbdoc");
    m.def("get_prediction", &get_prediction, R"pbdoc(
        Some help text here

        Some other explanation about the get_prediction function.
    )pbdoc");
    m.def("get_weightMatrix", &get_weightMatrix, R"pbdoc(
        Some help text here

        Some other explanation about the get_weightMatrix function.
    )pbdoc");
    m.def("getVI", &getVI, R"pbdoc(
        Some help text here
    
        Some other explanation about the getVI function.
    )pbdoc");
    m.def("getTreeNodeCount", &getTreeNodeCount, R"pbdoc(
        Some help text here

        Some other explanation about the getTreeNodeCount function.
    )pbdoc");
    m.def("get_path", &get_path, R"pbdoc(
        Some help text here

        Some other explanation about the get_path function.
    )pbdoc");
    m.def("py_reconstructree", &py_reconstructree_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the py_reconstructree function.
    )pbdoc");
    m.def("delete_forestry", &delete_forestry, R"pbdoc(
        Some help text here

        Some other explanation about the delete_forestry function.
    )pbdoc");
    m.def("predictOOB_forest", &predictOOB_forest_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the predictOOB_forest function.
    )pbdoc");
    m.def("predict_forest", &predict_forest_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the predict_forest function.
    )pbdoc");
    m.def("fill_tree_info", &fill_tree_info_wrapper, R"pbdoc(
        Some help text here

        Some other explanation about the fill_tree_info function.
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

