test_that("Tests hierarchical shrinkage works as expected", {
  context("check total node number equals length of full weights and count vector")

  set.seed(238943202)
  # Test example with simple step functio
  x <- iris[, -1]
  y <- iris[, 1]
  rf <- forestry(x,
                y,
                ntree = 10,
                maxDepth = 5)
  rf <- make_savable(rf)

  # Now check things are the right size
  var_id  = rf@R_forest[[1]]$var_id
  num_nodes = length(var_id[var_id>=0]) + length(var_id[var_id < 0])/2
  expect_equal(length(rf@R_forest[[1]]$average_count),num_nodes)
  expect_equal(length(rf@R_forest[[1]]$weightsFull),num_nodes)
})
