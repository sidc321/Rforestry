test_that("Tests that scaling works correctly", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('Test forest with/without scaling')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree=1,
    maxDepth = 2,
    seed=12
  )

  # Test forestry with scaling
  forest_scaled <- forestry(
    x,
    y,
    ntree=1,
    scale = TRUE,
    maxDepth = 2,
    seed=12
  )

  p <- predict(forest, newdata = x)
  p.scaled <- predict(forest_scaled, newdata = x)

  expect_equal(
    all.equal(p,p.scaled),
    TRUE
  )
  # plot(forest)
  # plot(forest_scaled)

})
