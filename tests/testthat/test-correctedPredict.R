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

})
