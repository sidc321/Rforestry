// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// rcpp_cppDataFrameInterface
SEXP rcpp_cppDataFrameInterface(Rcpp::List x, Rcpp::NumericVector y, Rcpp::NumericVector catCols, int numRows, int numColumns);
RcppExport SEXP _forestry_rcpp_cppDataFrameInterface(SEXP xSEXP, SEXP ySEXP, SEXP catColsSEXP, SEXP numRowsSEXP, SEXP numColumnsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type catCols(catColsSEXP);
    Rcpp::traits::input_parameter< int >::type numRows(numRowsSEXP);
    Rcpp::traits::input_parameter< int >::type numColumns(numColumnsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppDataFrameInterface(x, y, catCols, numRows, numColumns));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_cppBuildInterface
SEXP rcpp_cppBuildInterface(Rcpp::List x, Rcpp::NumericVector y, Rcpp::NumericVector catCols, int numRows, int numColumns, int ntree, bool replace, int sampsize, int mtry, float splitratio, int nodesizeSpl, int nodesizeAvg, int nodesizeStrictSpl, int nodesizeStrictAvg, int seed, int nthread, bool verbose, bool middleSplit, int maxObs, bool doubleTree, bool existing_dataframe_flag, SEXP existing_dataframe);
RcppExport SEXP _forestry_rcpp_cppBuildInterface(SEXP xSEXP, SEXP ySEXP, SEXP catColsSEXP, SEXP numRowsSEXP, SEXP numColumnsSEXP, SEXP ntreeSEXP, SEXP replaceSEXP, SEXP sampsizeSEXP, SEXP mtrySEXP, SEXP splitratioSEXP, SEXP nodesizeSplSEXP, SEXP nodesizeAvgSEXP, SEXP nodesizeStrictSplSEXP, SEXP nodesizeStrictAvgSEXP, SEXP seedSEXP, SEXP nthreadSEXP, SEXP verboseSEXP, SEXP middleSplitSEXP, SEXP maxObsSEXP, SEXP doubleTreeSEXP, SEXP existing_dataframe_flagSEXP, SEXP existing_dataframeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type catCols(catColsSEXP);
    Rcpp::traits::input_parameter< int >::type numRows(numRowsSEXP);
    Rcpp::traits::input_parameter< int >::type numColumns(numColumnsSEXP);
    Rcpp::traits::input_parameter< int >::type ntree(ntreeSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< int >::type sampsize(sampsizeSEXP);
    Rcpp::traits::input_parameter< int >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< float >::type splitratio(splitratioSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeSpl(nodesizeSplSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeAvg(nodesizeAvgSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeStrictSpl(nodesizeStrictSplSEXP);
    Rcpp::traits::input_parameter< int >::type nodesizeStrictAvg(nodesizeStrictAvgSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type nthread(nthreadSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type middleSplit(middleSplitSEXP);
    Rcpp::traits::input_parameter< int >::type maxObs(maxObsSEXP);
    Rcpp::traits::input_parameter< bool >::type doubleTree(doubleTreeSEXP);
    Rcpp::traits::input_parameter< bool >::type existing_dataframe_flag(existing_dataframe_flagSEXP);
    Rcpp::traits::input_parameter< SEXP >::type existing_dataframe(existing_dataframeSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppBuildInterface(x, y, catCols, numRows, numColumns, ntree, replace, sampsize, mtry, splitratio, nodesizeSpl, nodesizeAvg, nodesizeStrictSpl, nodesizeStrictAvg, seed, nthread, verbose, middleSplit, maxObs, doubleTree, existing_dataframe_flag, existing_dataframe));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_cppPredictInterface
Rcpp::List rcpp_cppPredictInterface(SEXP forest, Rcpp::List x, std::string aggregation);
RcppExport SEXP _forestry_rcpp_cppPredictInterface(SEXP forestSEXP, SEXP xSEXP, SEXP aggregationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type x(xSEXP);
    Rcpp::traits::input_parameter< std::string >::type aggregation(aggregationSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_cppPredictInterface(forest, x, aggregation));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_OBBPredictInterface
float rcpp_OBBPredictInterface(SEXP forest);
RcppExport SEXP _forestry_rcpp_OBBPredictInterface(SEXP forestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_OBBPredictInterface(forest));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_getObservationSizeInterface
float rcpp_getObservationSizeInterface(SEXP df);
RcppExport SEXP _forestry_rcpp_getObservationSizeInterface(SEXP dfSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type df(dfSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_getObservationSizeInterface(df));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_AddTreeInterface
void rcpp_AddTreeInterface(SEXP forest, int ntree);
RcppExport SEXP _forestry_rcpp_AddTreeInterface(SEXP forestSEXP, SEXP ntreeSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    Rcpp::traits::input_parameter< int >::type ntree(ntreeSEXP);
    rcpp_AddTreeInterface(forest, ntree);
    return R_NilValue;
END_RCPP
}
// rcpp_CppToR_translator
Rcpp::List rcpp_CppToR_translator(SEXP forest);
RcppExport SEXP _forestry_rcpp_CppToR_translator(SEXP forestSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type forest(forestSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_CppToR_translator(forest));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_forestry_rcpp_cppDataFrameInterface", (DL_FUNC) &_forestry_rcpp_cppDataFrameInterface, 5},
    {"_forestry_rcpp_cppBuildInterface", (DL_FUNC) &_forestry_rcpp_cppBuildInterface, 22},
    {"_forestry_rcpp_cppPredictInterface", (DL_FUNC) &_forestry_rcpp_cppPredictInterface, 3},
    {"_forestry_rcpp_OBBPredictInterface", (DL_FUNC) &_forestry_rcpp_OBBPredictInterface, 1},
    {"_forestry_rcpp_getObservationSizeInterface", (DL_FUNC) &_forestry_rcpp_getObservationSizeInterface, 1},
    {"_forestry_rcpp_AddTreeInterface", (DL_FUNC) &_forestry_rcpp_AddTreeInterface, 2},
    {"_forestry_rcpp_CppToR_translator", (DL_FUNC) &_forestry_rcpp_CppToR_translator, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_forestry(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
