test_that("Tests that the predictInfo function works", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Test getting the weightMatrix and observations used')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    OOBhonest = TRUE,
    seed = 2312
  )

  info <- predictInfo(forest, x, aggregation = "average")

  info.oob <- predictInfo(forest, x, aggregation = "oob")

  info.double <- predictInfo(forest, x, aggregation = "doubleOOB")

  expect_equal(nrow(info$weightMatrix), 150)

  first_ob <- info$obsInfo[[1]]

  skip_if_not_mac()
  expect_equal(all.equal(info$obsInfo[[1]]$Weight[1:10],
                         c(0.068042104, 0.015729129, 0.008917891, 0.008753721,
                           0.053599199, 0.008646022, 0.030503379, 0.022268962,
                           0.012211569, 0.005780545)),TRUE)

})
