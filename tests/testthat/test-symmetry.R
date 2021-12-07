test_that("Tests symmetry + monotonicity + missing data + OOBhonest + Monotone Avg", {

  library(Rforestry)
  context("1-dimensional example")

  set.seed(23322)
  n <- 1000
  x <- matrix(runif(n,min=-2,max=2), ncol=1)
  y <- x[,1]**3
  colnames(x) <- c("V1")
  # colnames(x) <- c("V1","V2")
  # plot(x[,1],y)
  # x[135:235,1] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=212342,
                 #maxDepth = 3,
                 #mtry=2,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(1),
                 monotoneAvg = TRUE,
                 symmetric = c(1))

  p <- predict(rf, newdata = x)
  # plot(x[,1],p)

  context("2-dimensional example")

  set.seed(23322)
  n <- 1000
  x <- matrix(runif(2*n,min=-2,max=2), ncol=2)
  y <- x[,2]**3
  colnames(x) <- c("V1","V2")
  # plot(x[,1],y)
  # x[135:235,1] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=212342,
                 #maxDepth = 3,
                 #mtry=2,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(0,1),
                 monotoneAvg = TRUE,
                 symmetric = c(0,1))

  p <- predict(rf, newdata = x)
  # plot(x[,2],p)

  context("2-dimensional example in second feature")

  set.seed(23322)
  n <- 1000
  x <- matrix(runif(2*n,min=-2,max=2), ncol=2)
  y <- x[,1]**3
  colnames(x) <- c("V1","V2")
  # plot(x[,2],y)
  # x[135:235,1] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=212342,
                 #maxDepth = 3,
                 #mtry=2,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(1,0),
                 monotoneAvg = TRUE,
                 symmetric = c(1,0))

  p <- predict(rf, newdata = x)
  # plot(x[,1],p)

  # Make synthetic data set with V2 fixed
  # x_new <- data.frame(V1 = seq(-2,2,length.out = 1000), V2 = rep(.2,n))
  # p_new <- predict(rf, newdata = x_new)
  # plot(x_new[,1],p_new)

  #
  # plot(x[,1], y)

  # Test with Iris now

  context('Tests symmetric = TRUE flag with monotonicity + NAs + OOBhonesty')
  # Try with some missing data ---------------------------------------------------
  for (seed_i in 1:10) {
    set.seed(seed_i)

    # Generate Data
    x <- matrix(runif(n,min=-2,max=2), ncol=1)
    y <- x[,1]**3
    colnames(x) <- c("V1")

    x_missing <- sample(1:n, size = round(.2*n), replace = FALSE)
    x[x_missing,1] <- NA

    # Regress
    rf <- forestry(x=x,
                   y=y,
                   ntree=500,
                   seed=seed_i,
                   OOBhonest = TRUE,
                   monotonicConstraints = c(1),
                   monotoneAvg = TRUE,
                   symmetric = c(1),
                   scale = FALSE
    )

    # predict
    preds_na <- predict(rf, newdata = x)

    # plot(x[,1], y)
    # plot(x[,1], preds_na)
  }
  expect_equal(length(preds_na),n)

  #plot(x[,1],p)
  context('Tests symmetric = TRUE flag with monotonicity + NAs + high dimensional X + OOBhonesty')

  # Try with some missing data in many columns -----------------------------------
  for (seed_i in 1:10) {
    set.seed(seed_i)
    p <- 10
    # Generate Data
    x <- matrix(runif(n*p,min=-2,max=2), ncol=p)
    y <- x[,1]**3
    colnames(x) <- c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10")

    for (i in 1:5) {
      x_missing <- sample(1:n, size = round(.2*n), replace = FALSE)
      x[x_missing,i] <- NA
    }

    # Regress
    rf <- forestry(x=x,
                   y=y,
                   ntree=500,
                   seed=seed_i,
                   OOBhonest = TRUE,
                   scale = FALSE,
                   monotonicConstraints = c(1,1,-1,rep(0,7)),
                   monotoneAvg = TRUE
    )

    # predict
    preds_p_na <- predict(rf, newdata = x)
  }
  expect_equal(length(preds_p_na), n)

  # Test the predictions -------------------------------------------------------
  set.seed(23322)

  n <- 1000
  x <- matrix(runif(n,min=-2,max=2), ncol=1)
  y <- x[,1]**3
  colnames(x) <- c("V1")

  x_missing <- sample(1:n, size = round(.2*n), replace = FALSE)
  x[x_missing,1] <- NA

  # Regress
  rf <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=seed_i,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(1),
                 monotoneAvg = TRUE,
                 symmetric = c(1)
  )

  # predict
  preds_na <- predict(rf, newdata = x)

  # Regress
  rf_old <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=seed_i,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(1),
                 monotoneAvg = TRUE,
                 symmetric = c(0)
  )

  # predict
  preds_old <- predict(rf_old, newdata = x)

  # Look at MSE's
  mse_sym <- mean((preds_na-y)**2)
  mse_std <- mean((preds_old-y)**2)

  expect_equal(mse_sym, 7.5058636474109, tolerance = 1e-4)
  expect_equal(mse_std, 2.16613136977747, tolerance = 1e-4)


  # Test that predictions follow monotone constraints
  new_x <- data.frame(V1 = seq(from=-1.5,to=1.5,length.out=50))
  monotone_preds <- predict(rf, newdata = new_x, seed=239)
  expect_equal(all.equal(order(monotone_preds), 1:50),TRUE)

  context("Test that NA behavior is working correctly with symmetry")
  # Test that NA behavior is working as expected -------------------------------
  set.seed(2332)
  x <- data.frame(V1 = runif(100,min=-1,max=1))
  y <- ifelse(x > .5,1,ifelse(x < -.5,-1,0))

  # plot(x[,1], y)

  x$V1[which(x$V1 > .1 & x$V1 <.25)] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=1,
                 seed=23,
                 scale = FALSE,
                 symmetric = c(1),
                 maxDepth = 1)

  preds_sym <- predict(rf, newdata = x)
  weights <- unique(preds_sym)

  preds_na <- predict(rf, newdata = data.frame(V1 = rep(NA,5)))
  #expect_equal(all.equal(preds_na, rep(median(weights),5)), TRUE)


  x <- data.frame(V1 = rnorm(100))
  y <- (x$V1)*2

  set.seed(275)
  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1,
                 monotonicConstraints = c(1),
                 seed = 298,
                 symmetric = c(1),
                 scale = FALSE,
                 ntree = 1)

  preds <- predict(rf, newdata = x)

  # Now add missings, should all go to the right
  x$V1[which(x$V1 > 1.3)] <- NA

  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1,
                 seed = 298,
                 monotonicConstraints = c(1),
                 symmetric = c(1),
                 scale = FALSE,
                 ntree = 1)

  preds_overall <- predict(rf, newdata = x)
  preds_na <- predict(rf, newdata = data.frame(V1 = c(NA,NA,NA)))

  #expect_equal(all.equal(preds_na, rep(max(preds_overall),3)),TRUE)

  # Test the code that was crashing before
  set.seed(1)
  n <- 11257
  y <- rnorm(n)
  x <- matrix(rnorm(2*n), ncol = 2)

  rf <- forestry(
    x=x,
    y=y,
    monotonicConstraints = c(-1,0),
    monotoneAvg = TRUE,
    ntree=1000,
    scale = FALSE,
    OOBhonest = TRUE,
    symmetric = c(1,0)
  )

  context("Test missing data with several other features")
  #p <- predict(rf, newdata = x)
  #expect_equal(length(p), n)
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

  preds <- predict(rf, newdata = data.frame(x = rep(NA,10)))

  expect_equal(all.equal(preds, rep(0.9832234,10),tolerance = 1e-5),TRUE)

  # NOW do again with symmetric ================================================
  set.seed(382)
  # First example we can test three different regions
  x <- rnorm(100)
  y <- ifelse(x > 1,1, ifelse(x < -1,-1,0)) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)

  # Only make right observations missing now
  missing_idx <- sample(which(x$x > 1), size = 10, replace = FALSE)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 monotonicConstraints = c(1),
                 symmetric = c(1),
                 scale=FALSE,
                 seed = 2323,
                 ntree=1,
                 OOBhonest = TRUE,
                 monotoneAvg = TRUE,
                 maxDepth = 1)

  preds3 <- predict(rf, newdata = data.frame(x = rep(NA,5)))
  p_all <- predict(rf, newdata=x)
  expect_equal(length(preds3),5)
  #plot(rf)


  # Try simon's example
  set.seed(1)
  n <- 11257
  y <- rnorm(n)
  x <- matrix(rnorm(2*n),ncol=2)

  rf <- forestry(
    x=x,
    y=y,
    monotonicConstraints = c(-1,0),
    monotoneAvg = TRUE,
    scale = FALSE,
    ntree=1000,
    OOBhonest = TRUE,
    symmetric = c(1,0)
  )

  preds2 <- predict(rf, newdata = x)
  expect_equal(length(preds2), n)
  # Some problems:
  # Predicting with NA's is still random when symmetric  = TRUE
  # Some very weird interaction of honesty and monotoneAVG
  # Just with OOB honest the predictions are NA


  # Test example that was crashing before
  set.seed(382)
  # First example we can test three different regions
  x <- rnorm(100)
  y <- ifelse(x > 1,1, ifelse(x < -1,-1,0)) + rnorm(100, mean = 0, sd = .1)
  x <- data.frame(x)

  # plot(x$x, y)

  # Only make right observations missing now
  missing_idx <- sample(which(x$x > 1), size = 10, replace = FALSE)
  x$x[missing_idx] <- NA

  rf <- forestry(x = x,
                 y = y,
                 monotonicConstraints = c(1),
                 symmetric = c(1),
                 scale=FALSE,
                 seed = 2323,
                 ntree=1,
                 OOBhonest = TRUE,
                 monotoneAvg = TRUE,
                 maxDepth = 1)

  p <- predict(rf, newdata = data.frame(x = rep(NA,5)))
  p_all <- predict(rf, newdata=x)

  expect_equal(length(p), 5)
  expect_equal(length(p_all), nrow(x))


})
