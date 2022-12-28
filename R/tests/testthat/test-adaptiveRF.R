test_that("Tests that adaptive Forestry works", {
  set.seed(292313)

  test_idx <- sample(nrow(iris), 11)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  context("Train adaptiveForestry")
  rf <- adaptiveForestry(x = x_train,
                          y = y_train,
                          ntree.first = 25,
                          ntree.second = 500,
                          nthread = 2)
  p <- predict(rf@second.forest, x_test)

  expect_equal(length(p), 11)

  context("High precision test for prediction of adaptiveForestry")
  skip_if_not_mac()

  #expect_equal(all.equal(p, c(4.830888889, 5.568333333, 7.060666667, 6.382533333,
  #                            6.701857143, 5.317333333, 6.158300000, 5.774000000,
  #                            5.671666667, 4.766479853, 4.917466667), tolerance = 1e-6), TRUE)

})
