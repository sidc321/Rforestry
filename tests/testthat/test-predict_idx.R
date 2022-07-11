test_that("Tests predict index option", {
  context('Tests getting specific predictions for one observation')

  x <- iris[1:40,-c(1,5)]
  y <- iris[1:40,1]

  # Given a
  test_tree_preds <- function(rf) {
    rf <- make_savable(rf)
    # Check first tree by hand
    insample_idx <- sort(union(rf@R_forest[[1]]$averagingSampleIndex,
                               rf@R_forest[[1]]$splittingSampleIndex))
    outsample_idx <- setdiff(1:nrow(rf@processed_dta$processed_x), insample_idx)

    p_in <- predict(rf, newdata = rf@processed_dta$processed_x[1,], predictIdx = insample_idx)
    expect_equal(p_in, NaN)
    p_out <- predict(rf, newdata = rf@processed_dta$processed_x[1,], predictIdx = outsample_idx)
    expect_gt(p_out, 1)

    p_all <- predict(rf, newdata = rf@processed_dta$processed_x[1,], predictIdx = c(outsample_idx,
                                                                                    insample_idx))
    expect_equal(p_all, NaN)
  }

  test_forest_preds <- function(rf) {
    # First check normal predictions
    pred_all <- predict(rf, newdata = rf@processed_dta$processed_x[1,], weightMatrix = TRUE)
    expect_equal(sum(pred_all$weightMatrix[1,]), 1)

    pred_holdout <- predict(rf, newdata = rf@processed_dta$processed_x[1,], weightMatrix = TRUE, predictIdx = c(1:4))
    expect_equal(sum(pred_holdout$weightMatrix[1,1:4]), 0)

    # Now see if a prediction was able to be made
    if (is.nan(pred_holdout$predictions)) {
      expect_equal(sum(pred_holdout$weightMatrix[1,]), 0)
    } else {
      expect_equal(sum(pred_holdout$weightMatrix[1,]), 1)
    }
  }

  rf <- forestry(x = x,
                 y = y,
                 seed = 131,
                 OOBhonest = TRUE,
                 maxDepth = 2,
                 scale = FALSE,
                 ntree = 1)
  test_tree_preds(rf)

  # Test whole forest
  rf <- forestry(x = x,
                 y = y,
                 seed = 131,
                 OOBhonest = TRUE,
                 maxDepth = 2,
                 scale = FALSE,
                 ntree = 1000)
  test_forest_preds(rf)




})
