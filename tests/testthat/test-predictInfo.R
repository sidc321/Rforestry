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
                         c(0.065719889164469164066773, 0.016520037897421121358965, 0.008480538754331324105551,
                           0.007944978989019234674740, 0.052003063531344574654813, 0.008815330945284504879367,
                           0.033636028293209586925716, 0.024207273402751239982367,
                           0.011876750142739506133083, 0.005217297104913203205367), tolerance = 1e-8),TRUE)

})
