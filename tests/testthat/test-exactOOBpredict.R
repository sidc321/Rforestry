test_that("Tests that exact works with different aggregation types", {
  x <- data.frame(X1 = rnorm(1e3), X2 = rnorm(1e3))
  y <- rnorm(1e3)

  context('Test that exact predict with oob aggregation works')
  # Set seed for reproductivity
  set.seed(24750371)

  rf <- forestry(x=x,
                 y=y,
                 OOBhonest = TRUE,
                 ntree = 500,
                 seed = 1)


  skip_if_not_mac()
  for (i in 1:2) {
    p1 <- predict(rf, x)
    p2 <- predict(rf, x)
    expect_equal(all.equal(p1,p2, tolerance = 1e-10), TRUE)
  }

  for (i in 1:3) {
    p1 <- predict(rf, x, aggregation = "oob")
    p2 <- predict(rf, x, aggregation = "oob")
    expect_equal(all.equal(p1,p2, tolerance = 1e-10), TRUE)
  }

  for (i in 1:3) {
    p1 <- predict(rf, x, aggregation = "doubleOOB")
    p2 <- predict(rf, x, aggregation = "doubleOOB")
    expect_equal(all.equal(p1,p2, tolerance = 1e-10), TRUE)
  }

})
