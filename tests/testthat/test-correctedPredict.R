test_that('Bias corrections', {

  context("Bias Corrected predictions")

  set.seed(121235312)
  n <- 1000
  p <- 100
  x <- matrix(rnorm(n * p), ncol = p)
  beta <- runif(p,min = 0, max = 1)
  y <- as.matrix(x) %*% beta + rnorm(1000)


  forest <- forestry(x =x,
                     y = y[,1],
                     OOBhonest = TRUE,
                     doubleBootstrap = TRUE)

  preds <- predict(forest, newdata = x, aggregation = "oob")

  rmse <- sqrt(mean((preds - y[,1])^2))

  # Now do some bias corrected predictions with the forest ---------------------
  pred.bc2 <- correctedPredict(forest,
                               newdata = x,
                               nrounds = 5)

  rmse.bc2 <- sqrt(mean((pred.bc2$test.preds - y[,1])^2))

  expect_equal(length(pred.bc2$test.preds), 1000)


  expect_gt(rmse.bc2,rmse)

  # Try the iris data ----------------------------------------------------------

  set.seed(1323)
  sample_idx <- sample(1:150, size = 50)
  sample_idx <- sort(sample_idx)

  x_test <- iris[sample_idx,-1]
  x_train <- iris[!(1:150 %in% sample_idx),-1]

  y_test <- iris[sample_idx,1]
  y_train <- iris[!(1:150 %in% sample_idx),1]

  forest <- forestry(x = x_train,
                     y = y_train,
                     scale = TRUE,
                     OOBhonest = TRUE)

  # Do bias correcitons on the new data
  pred.bc3 <- correctedPredict(forest,
                               newdata = x_test,
                              nrounds = 1)
  pred.bc3

  rmse.bc3 <- sqrt(mean((pred.bc3$test.preds - y)^2))





  # Now try the predict on an out of sample design matrix ----------------------
  x_new <- matrix(rnorm(n * p), ncol = p)
  y_new <- as.matrix(x) %*% beta



  context("Test passing forestry parameters to the bias correction")

  set.seed(121235312)
  n <- 100
  p <- 10
  x <- matrix(rnorm(n * p), ncol = p)
  beta <- runif(p,min = 1, max = 2)
  beta[6:10] <- 0
  y <- as.matrix(x) %*% beta
  x <- as.data.frame(x)


  rf <- forestry(x =x,
                 y = y[,1],
                 OOBhonest = TRUE,
                 doubleBootstrap = TRUE)

  params <- list(ntree = 1000,
                 groups=as.factor(c(rep(1,50), rep(2,50))),
                 minTreesPerFold = 400,
                 monotonicConstraints = c(rep(1,2),rep(0,3),1),
                 seed = 12312
                 )

  preds.bc <- correctedPredict(rf,
                               newdata = x,
                               feats = c(1:5),
                               params.forestry = params,
                               nrounds = 3)


  # Now check the trained forests and make sure they get the right parameters
  preds.bc2 <- correctedPredict(rf,
                               newdata = x,
                               feats = c(1:5),
                               params.forestry = params,
                               keep_fits = TRUE,
                               nrounds = 3)

  context("Check the parameters of the fitted forests")
  for (fit_i in 1:3) {
    rf_i <- preds.bc2$fits[[fit_i]]

    # Make sure parameters are the same as in the params list
    expect_equal(rf_i@ntree, params$ntree)
    expect_equal(rf_i@minTreesPerFold, params$minTreesPerFold)
    expect_equal(all.equal(rf_i@monotonicConstraints, params$monotonicConstraints),TRUE)
    expect_equal(all.equal(as.factor(rf_i@groups), params$groups),TRUE)
  }


  context("Try passing bad parameters to params.forestry")
  # Now if we use the old params but don't use adaptiveForestry,
  # the ntree.first ntree.second parameters should be invalid

  params <- list(ntree.first = 250, ntree.second = 500, seed = 1397)

  expect_error(
    preds.bc4 <- correctedPredict(rf,
                                  newdata = x,
                                  feats = c(1:5),
                                  params.forestry = params,
                                  nrounds = 1),
    "Invalid parameter in params.forestry: ntree.first Invalid parameter in params.forestry: ntree.second "
  )


})
