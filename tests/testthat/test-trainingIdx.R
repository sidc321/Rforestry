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
    expect_equal(all.equal(p_oob[1:10], p_oob_idx), TRUE)

    # Check weight matrices are equal
    p_oob_weights <- predict(rf, newdata = x_new, aggregation = "oob", weightMatrix = TRUE)
    p_oob_idx_weights <- predict(rf, newdata = x_new[1:10,], aggregation = "oob", trainingIdx = 1:10, weightMatrix = TRUE)

    expect_equal(all.equal((p_oob_weights$weightMatrix %*% as.matrix(rf@processed_dta$y))[,1][1:10],
                           (p_oob_idx_weights$weightMatrix %*% as.matrix(rf@processed_dta$y))[,1])
                 , TRUE)

    # Test doubleOOB aggregation
    if (rf@doubleBootstrap) {
      p_doob <- predict(rf, newdata = x_new, aggregation = "doubleOOB")
      p_doob_idx <- predict(rf, newdata = x_new[1:10,], aggregation = "doubleOOB", trainingIdx = 1:10)
      expect_equal(all.equal(p_doob[1:10], p_doob_idx[1:10]), TRUE)

      # Check weight matrix equality
      p_doob_weights <- predict(rf, newdata = x_new, aggregation = "doubleOOB", weightMatrix = TRUE)
      p_doob_idx_weights <- predict(rf, newdata = x_new[1:10,], aggregation = "doubleOOB", trainingIdx = 1:10, weightMatrix = TRUE)

      expect_equal(all.equal((p_doob_weights$weightMatrix %*% as.matrix(rf@processed_dta$y))[,1][1:10],
                             (p_doob_idx_weights$weightMatrix %*% as.matrix(rf@processed_dta$y))[,1])
                   , TRUE)
    }
  }

  xtrain <- iris[,-c(1,5)]
  ytrain <- iris[,1]

  # Check standard OOB honest forest
  forest <- forestry(x = xtrain,
                     y = ytrain,
                     OOBhonest = TRUE)

  #predict(forest, newdata = xtrain[c(1,4,5),], trainingIdx = c(1,4,5), aggregation = "oob", weightMatrix = TRUE)

  check_oob_preds(forest)

  # Check standard forest
  forest_other <- forestry(x = xtrain,
                           y = ytrain)

  check_oob_preds(forest_other)


  # Check standard honest forest
  forest_std_honesty <- forestry(x = xtrain,
                                 y = ytrain,
                                 splitratio = .4)

  check_oob_preds(forest_std_honesty)

  # Check OOBhonest = TRUE with no double Bootstrap forest
  forest_no_double_boot <- forestry(x = xtrain,
                                    y = ytrain,
                                    OOBhonest = TRUE,
                                    doubleBootstrap = FALSE)

  check_oob_preds(forest_no_double_boot)

  # Test groups option forest
  forest_groups <- forestry(x = xtrain,
                            y = ytrain,
                            OOBhonest = TRUE,
                            groups = as.factor(iris$Species))

  check_oob_preds(forest_groups)

  # Check error handling =======================================================
  expect_error(
    predict_OOBpreds <- predict(forest_groups, aggregation = "average"),
    "When using an aggregation that is not oob or doubleOOB, one must supply newdata"
  )

  expect_error(
    predict_OOBpreds <- predict(forest_groups, aggregation = "oob", newdata = xtrain[1:30,]),
    "trainingIdx must be set when doing out of bag predictions with a data set not equal in size to the training data set"
  )

  expect_error(
    predict_OOBpreds <- predict(forest_groups, aggregation = "oob", newdata = xtrain[1:30,], trainingIdx = 1:20),
    "The length of trainingIdx must be the same as the number of observations in the training data"
  )

})
