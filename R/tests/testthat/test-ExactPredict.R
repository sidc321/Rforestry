test_that("Tests that exact prediction flag is working", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Test that exact and inexact predictions match')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y
  )

  skip_if_not_mac()

  context("Test that exact predictions work with aggregation = oob and doubleOOB")

  rf <- forestry(x,y,OOBhonest = TRUE)

  p1 <- predict(rf, aggregation = "oob")
  p2 <- predict(rf, aggregation = "oob")
  expect_equal(all.equal(p1,p2,tolerance = 1e-12),TRUE)

  p3 <- predict(rf, aggregation = "doubleOOB")
  p4 <- predict(rf, aggregation = "doubleOOB")
  expect_equal(all.equal(p1,p2,tolerance = 1e-12),TRUE)

  # Test predict
  exact_predictions <- predict(forest,
                               x[c(5, 100, 104,105),],
                               exact = TRUE,
                               aggregation = 'weightMatrix')
  inexact_predictions <- predict(forest,
                                 x[c(5, 100, 104, 105),],
                                 exact = FALSE,
                                 aggregation = 'weightMatrix')

  context("Check matching predictions")
  expect_equal(all.equal(exact_predictions$predictions,
                         inexact_predictions$predictions, tolerance = 1e-3),
               TRUE)
  context("Check matching weightMatrix")
  expect_equal(all.equal(exact_predictions$weightMatrix[3,],
                         inexact_predictions$weightMatrix[3,], tolerance = 1e-3),
               TRUE)

})
