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

  rmse <- sqrt(mean((preds - y)^2))

  extremes <- c(range(preds), range(y))
  # We can see it is very regularized here
  # library(ggplot2)
  # ggplot(aes(x =X, y=Y), data=data.frame(Y = y, X=preds))+
  #   geom_point()+
  #   geom_abline(intercept = 0, slope = 1)+
  #   ylim(min(extremes), max(extremes))+
  #   xlim(min(extremes), max(extremes))+
  #   labs(y = "True Y", x = "Predicted Y")

  # Now do bias corrected predictions

  # Run many different settings for the bias correction
  pred.bc <- correctedPredict(forest,
                              newdata = x,
                              simple = FALSE,
                              nrounds = 0)

  rmse.bc <- sqrt(mean((pred.bc - y)^2))

  pred.bc <- correctedPredict(forest,
                              newdata = x,
                              nrounds = 1)
  rmse.bc <- sqrt(mean((pred.bc - y)^2))

  pred.bc <- correctedPredict(forest,
                              newdata = x,
                              monotone = TRUE,
                              simple=FALSE,
                              nrounds = 1)
  rmse.bc <- sqrt(mean((pred.bc - y)^2))

  pred.bc <- correctedPredict(forest,
                              nrounds = 0)


  rmse.bc <- sqrt(mean((pred.bc - y)^2))

  expect_equal(length(pred.bc), 1000)


  expect_gt(rmse, rmse.bc)
  extremes <- c(range(pred.bc), range(y))

  # We can see it is very regularized here
  # library(ggplot2)
  # ggplot(aes(x =X, y=Y), data=data.frame(Y = y, X=pred.bc))+
  #   geom_point()+
  #   geom_abline(intercept = 0, slope = 1)+
  #   ylim(min(extremes), max(extremes))+
  #   xlim(min(extremes), max(extremes))+
  #   labs(y = "True Y", x = "Predicted Y")

  # Now do some bias corrected predictions with the forest ---------------------
  pred.bc2 <- correctedPredict(forest,
                               nrounds = 5)

  rmse.bc2 <- sqrt(mean((pred.bc2 - y)^2))

  expect_equal(length(pred.bc2), 1000)


  expect_gt(rmse, rmse.bc2)
  extremes <- c(range(pred.bc2), range(y))

  # Finally do some bias corrected predictions with monotone prediction --------
  pred.bc3 <- correctedPredict(forest,
                               nrounds = 5,
                               monotone = TRUE)

  rmse.bc3 <- sqrt(mean((pred.bc3 - y)^2))

  expect_equal(length(pred.bc3), 1000)


  expect_gt(rmse, rmse.bc3)


  # Now try the predict on an out of sample design matrix ----------------------
  x_new <- matrix(rnorm(n * p), ncol = p)
  y_new <- as.matrix(x) %*% beta



  pred.bc <- correctedPredict(forest,
                              newdata = x_new,
                              simple = FALSE,
                              nrounds = 0)

  rmse.bc <- sqrt(mean((pred.bc - y_new)^2))

  # extremes <- c(range(pred.bc), range(y_new))
  # library(ggplot2)
  # ggplot(aes(x =X, y=Y), data=data.frame(Y = y, X=pred.bc))+
  #   geom_point()+
  #   geom_abline(intercept = 0, slope = 1)+
  #   ylim(min(extremes), max(extremes))+
  #   xlim(min(extremes), max(extremes))+
  #   labs(y = "True Y", x = "Predicted Y")

  pred.bc <- correctedPredict(forest,
                              newdata = x_new,
                              nrounds = 1)
  rmse.bc <- sqrt(mean((pred.bc - y_new)^2))

  pred.bc <- correctedPredict(forest,
                              newdata = x_new,
                              monotone = TRUE,
                              simple= FALSE,
                              nrounds = 1)
  rmse.bc <- sqrt(mean((pred.bc - y_new)^2))

  pred.bc <- correctedPredict(forest,
                              newdata = x_new,
                              nrounds = 0)

  rmse.bc <- sqrt(mean((pred.bc - y_new)^2))




  pred.bcnew <- correctedPredict(forest,
                                 newdata = x_new,
                                 double = TRUE,
                                 nrounds = 0)

  rmse.bcnew <- sqrt(mean((pred.bcnew - y_new)^2))

  pred.old <- predict(forest,
                      newdata = x_new)

  rmse.old <- sqrt(mean((pred.old - y_new)^2))


  #expect_gt(rmse.old, rmse.bcnew)


  # We can see it is very regularized here
  # library(ggplot2)
  # ggplot(aes(x =X, y=Y), data=data.frame(Y = y, X=pred.bc2))+
  #   geom_point()+
  #   geom_abline(intercept = 0, slope = 1)+
  #   ylim(min(extremes), max(extremes))+
  #   xlim(min(extremes), max(extremes))+
  #   labs(y = "True Y", x = "Predicted Y")

  # library(dbarts)
  # bart_fit <- bart(x.train = x,
  #                  y.train = y,
  #                  keeptrees = TRUE,
  #                  verbose = FALSE)
  #
  # p_bart <- bart_pred_func(bart_fit, x)
  #
  # rmse.bart <- sqrt(mean((p_bart - y)^2))

  context("Test passing some features to the bias correction")

  set.seed(121235312)
  n <- 100
  p <- 100
  x <- matrix(rnorm(n * p), ncol = p)
  beta <- runif(p,min = 0, max = 1)
  y <- as.matrix(x) %*% beta + rnorm(100)


  forest <- forestry(x =x,
                     y = y[,1],
                     OOBhonest = TRUE,
                     doubleBootstrap = TRUE)

  # Do out of sample bias correction
  preds <- correctedPredict(forest,
                            newdata = x[1:25,],
                            feats = c(78),
                            simple = TRUE,
                            nrounds = 1)
  expect_equal(length(preds), 25)

  # In sample bias correction
  preds <- correctedPredict(forest,
                            feats = c(1,2,3,78),
                            simple = TRUE,
                            nrounds = 1)
  expect_equal(length(preds), n)


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
                 minTreesPerGroup = 400,
                 monotonicConstraints = c(rep(1,2),rep(0,3),1),
                 seed = 12312
                 )

  preds.bc <- correctedPredict(rf,
                               newdata = x,
                               feats = c(1:5),
                               params.forestry = params,
                               nrounds = 3)

  expect_equal(length(preds.bc), n)

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
    expect_equal(rf_i@minTreesPerGroup, params$minTreesPerGroup)
    expect_equal(all.equal(rf_i@monotonicConstraints, params$monotonicConstraints),TRUE)
    expect_equal(all.equal(as.factor(rf_i@groups), params$groups),TRUE)
  }

  context("Check adaptive forestry parameters")

  params <- list(ntree.first = 250, ntree.second = 500, seed = 1397)

  preds.bc3 <- correctedPredict(rf,
                                newdata = x,
                                feats = c(1:5),
                                params.forestry = params,
                                adaptive = TRUE,
                                nrounds = 3)
  expect_equal(length(preds.bc3), n)


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
