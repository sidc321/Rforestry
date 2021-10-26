test_that("Test missing data with several other features", {
  library(Rforestry)

  set.seed(382)
  # First example we can test three different regions
  x <- rnorm(100)
  y <- ifelse(x > 0, 1,0) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)

  # Only make right observations missing now
  missing_idx <- sample(which(x$x > 0), size = 10, replace = FALSE)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 seed=939,
                 monotonicConstraints = c(1),
                 maxDepth = 1)

  p <- predict(rf, newdata = data.frame(x = rep(NA,10)))

  expect_equal(all.equal(p, rep(0.9832234,10),tolerance = 1e-5),TRUE)

  # NOW do again with symmetric ================================================
  set.seed(382)
  # First example we can test three different regions
  x <- rnorm(100)
  y <- ifelse(x > 0,1, ifelse(x < -1,-1,0)) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)

  # Only make right observations missing now
  missing_idx <- sample(which(x$x > 0), size = 10, replace = FALSE)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 seed=939,
                 monotonicConstraints = c(1),
                 symmetric = TRUE,
                 #OOBhonest = TRUE,
                 #monotoneAvg = TRUE,
                 maxDepth = 1)

  #p <- predict(rf, newdata = x)

  #plot(rf)

  # Some problems:
  # Predicting with NA's is still random when symmetric  = TRUE
  # Some very weird interaction of honesty and monotoneAVG
  # Just with OOB honest the predictions are NA

})
