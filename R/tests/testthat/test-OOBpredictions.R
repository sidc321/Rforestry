test_that("Tests if OOB predictions are working correctly (normal setting)", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB Predictions')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  # Test OOB predictions
  expect_equal(mean((getOOBpreds(forest, noWarning = TRUE) -  iris[,1])^2), getOOB(forest), tolerance = 1e-5)

  skip_if_not_mac()

  expect_equal(all.equal(getOOBpreds(forest, noWarning = TRUE)[1:10], c(5.090343629238978095941, 4.663643797193019580050,
                                                                        4.651080538830537847161, 4.876606650517940622080,
                                                                        5.084682124151035154114, 5.346775151424274064027,
                                                                        5.064028401318675598475, 5.064491903453301802074,
                                                                        4.762799341542434561347, 4.790124445076102688290)), TRUE)
})


test_that("Tests if OOB predictions are working correctly (extreme setting)", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB Predictions extreme')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test a very extreme setting
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = FALSE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  expect_warning(
    testOOBpreds <- getOOBpreds(forest, noWarning = FALSE),
    "Samples are drawn without replacement and sample size is too big!"
  )

  expect_equal(testOOBpreds, NA, tolerance = 1e-4)
})
