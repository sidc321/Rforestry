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
                 #monotonicConstraints = c(0,1),
                 #monotoneAvg = TRUE,
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
  for (seed_i in 1:2) {
    n=100
    set.seed(seed_i)
    p <- 10
    # Generate Data
    x <- matrix(runif(n*p,min=-2,max=2), ncol=p)
    y <- x[,1]**3
    colnames(x) <- c("V1","V2","V3","V4","V5","V6","V7","V8","V9","V10")

    # Regress
    rf <- forestry(x=x,
                   y=y,
                   ntree=500,
                   seed=seed_i,
                   OOBhonest = TRUE,
                   scale = FALSE,
                   monotonicConstraints = c(1,1,-1,rep(0,7)),
                   monotoneAvg = TRUE,
                   symmetric = c(1,rep(0,9))
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


  # Test that predictions follow monotone constraints
  new_x <- data.frame(V1 = seq(from=-1.5,to=1.5,length.out=50))
  monotone_preds <- predict(rf, newdata = new_x, seed=239)
  expect_equal(all.equal(order(monotone_preds), 1:50),TRUE)

})
