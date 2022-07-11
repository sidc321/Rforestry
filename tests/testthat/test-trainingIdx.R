test_that("Tests using trainingIdx when doing OOB predictions on smaller data", {
  library(Rforestry)

  # Helper function for checking the equality of predictions
  check_oob_preds <- function(rf) {
    # Try running for a transformation of x_new
    x_new <- rf@processed_dta$processed_x
    x_new[, 1] <- .23*x_new[, 1]

    # Test OOB Aggregation
    p_oob <- predict(rf, newdata = x_new, aggregation = "oob")
    p_oob_idx <- predict(rf, newdata = x_new[1:10,], aggregation = "oob", trainingIdx = 1:10)
    expect_equal(all.equal(p_oob[1:10], p_oob_idx[1:10]), TRUE)

    # Test doubleOOB aggregation
    if (rf@doubleBootstrap) {
      p_doob <- predict(rf, newdata = x_new, aggregation = "doubleOOB")
      p_doob_idx <- predict(rf, newdata = x_new[1:10,], aggregation = "doubleOOB", trainingIdx = 1:10)
      expect_equal(all.equal(p_doob[1:10], p_doob_idx[1:10]), TRUE)
    }
  }

  xtrain <- iris[,-c(1,5)]
  ytrain <- iris[,1]

  forest <- forestry(x = xtrain,
                     y = ytrain,
                     OOBhonest = TRUE)

  check_oob_preds(forest)

  forest_other <- forestry(x = xtrain,
                           y = ytrain)

  check_oob_preds(forest_other)

  forest_std_honesty <- forestry(x = xtrain,
                                 y = ytrain,
                                 splitratio = .4)

  check_oob_preds(forest_std_honesty)

  forest_no_double_boot <- forestry(x = xtrain,
                                    y = ytrain,
                                    OOBhonest = TRUE,
                                    doubleBootstrap = FALSE)

  check_oob_preds(forest_no_double_boot)

  # Check error handling =======================================================
  expect_error(
    predict_OOBpreds <- predict(forest, aggregation = "average"),
    "When using an aggregation that is not oob or doubleOOB, one must supply newdata"
  )

  expect_error(
    predict_OOBpreds <- predict(rf, aggregation = "oob", newdata = xtrain[1:30,]),
    "trainingIdx must be set when doing out of bag predictions with a data set not equal in size to the training data set"
  )

  expect_error(
    predict_OOBpreds <- predict(rf, aggregation = "oob", newdata = xtrain[1:30,], trainingIdx = 1:20),
    "The length of trainingIdx must be the same as the number of observations in the training data"
  )

})
