test_that('Bias corrections', {

  context("Bias Corrected predictions")

  set.seed(121235312)
  n <- 1000
  p <- 100
  x <- matrix(rnorm(n * p), ncol = p)
  beta <- runif(p,min = 0, max = 1)
  y <- as.matrix(x) %*% beta + rnorm(1000)


  forest <- forestry(x =x,
                     y = y[,1])

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
  pred.bc <- correctedPredict(forest,
                              newdata = x,
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
                               newdata = x,
                               nrounds = 5)

  rmse.bc2 <- sqrt(mean((pred.bc2 - y)^2))

  expect_equal(length(pred.bc2), 1000)


  expect_gt(rmse, rmse.bc2)
  extremes <- c(range(pred.bc2), range(y))

  # Finally do some bias corrected predictions with monotone prediction --------
  pred.bc3 <- correctedPredict(forest,
                               newdata = x,
                               nrounds = 5,
                               monotone = TRUE)

  rmse.bc3 <- sqrt(mean((pred.bc3 - y)^2))

  expect_equal(length(pred.bc3), 1000)


  expect_gt(rmse, rmse.bc3)

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
