library(testthat)
test_that("Tests different metrics for permutation VI", {

  library(pROC)

  n=500

  logistic <- function(x) {
    return(1/(1+exp(-x)))
  }

  evalAUC <- function(truth, pred){
    roc_model <- roc(response = truth, predictor = as.numeric(pred),quiet=TRUE)
    idx <- tail(which(roc_model$sensitivities >= 0.99), 1)
    tnr_model <- roc_model$specificities[idx]
    return(round(c(roc_model$auc, tnr_model), 7))
  }

  set.seed(56)
  x <- data.frame(matrix(rnorm(n*5), ncol = 5))
  y <- rbinom(n, size = 1, prob= logistic(x[,1] + .4*x[,2] - 2.1*x[,3]))

  # Test forestry (mimic RF)
  forest <- forestry(x, y, ntree = 50, seed = 1)

  preds <- predict(forest, aggregation = "oob")

  auc.metrics <-evalAUC(truth = y, pred = preds)

  vi.mse <- getVI(forest, seed = 101, metric = "mse")
  skip_if_not_mac()
  expect_equal(all(order(-vi.mse)[1:3] %in% c(1:3)), TRUE)
  vi.auc <- getVI(forest, seed = 101, metric = "auc")
  skip_if_not_mac()
  expect_equal(all(order(-vi.auc)[1:3] %in% c(1:3)), TRUE)
  vi.tnr <- getVI(forest, seed = 101, metric = "tnr")
  skip_if_not_mac()
  expect_equal(all(order(-vi.tnr)[1:3] %in% c(1:3)), TRUE)


  expect_equal(length(vi.tnr),5)
  expect_equal(length(vi.mse),5)
  expect_equal(length(vi.auc),5)

  expect_gt(max(vi.tnr),0)
  expect_gt(max(vi.mse),0)
  expect_gt(max(vi.auc),0)

  # Now try running with different seeds and comparing means to SE's
  # results <- matrix(nrow = 100,ncol=5)
  # for (iter in 1:100) {
  #   print(iter)
  #   vi.iter = getVI(forest, seed = iter, metric = "auc")
  #   results[iter,] = vi.iter
  # }
  #
  # means = colMeans(results)
  # se = apply(results, MARGIN = 2, FUN = function(x){return(sd(x))})
  # which(abs(means) > 3*se)

})
