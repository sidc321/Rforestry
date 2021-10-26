test_that("Tests several combinations of missing data, monotonicity, and symmetric splits", {

  context('Tests symmetric = TRUE flag with NA')

  x <- data.frame(V1 = rnorm(100))
  y <- (x$V1)*2

  set.seed(275)
  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1,
                 monotonicConstraints = c(1),
                 seed = 298,
                 symmetric = TRUE,
                 ntree = 1)

  preds <- predict(rf, newdata = x)

  # Now add missings, should all go to the right
  x$V1[which(x$V1 > 1.3)] <- NA

  rf <- forestry(x = x,
                 y = y,
                 maxDepth = 1,
                 seed = 298,
                 monotonicConstraints = c(1),
                 symmetric = TRUE,
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
    OOBhonest = TRUE,
    symmetric = TRUE
  )

  #p <- predict(rf, newdata = x)
  #expect_equal(length(p), n)

})
