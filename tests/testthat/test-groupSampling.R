test_that("Tests sampling with groups", {
  context('Test leaving out groups on a small dataset ')


  x <- iris[,-1]
  y <- iris[,1]

  # Use the species as the group
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 minTreesPerGroup = 1)

  rf <- make_savable(rf)

  # Test that a tree has been grown leaving out each species
  # Note that because we sort by seed after training the forest
  # so the first trees in R_forest have left out the last group etc
  idx1 <- rf@R_forest[[1]]$splittingSampleIndex
  expect_equal(all(!((101:150) %in% idx1)), TRUE)

  idx2 <- rf@R_forest[[2]]$splittingSampleIndex
  expect_equal(all(!((51:100) %in% idx2)), TRUE)

  idx3 <- rf@R_forest[[3]]$splittingSampleIndex
  expect_equal(all(!((1:50) %in% idx3)), TRUE)


  # Use the species as the group + 10 trees
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 minTreesPerGroup = 10)

  rf <- make_savable(rf)


  expect_equal(length(rf@R_forest), 30)

  for ( i in 1:30) {
    idx <- rf@R_forest[[i]]$splittingSampleIndex

    if (i %in% 1:10) {
      expect_equal(all(!((101:150) %in% idx)), TRUE)
    } else if (i %in% 11:20) {
      expect_equal(all(!((51:100) %in% idx)), TRUE)
    } else {
      expect_equal(all(!((1:50) %in% idx)), TRUE)
    }
  }


  # Test when using honesty, this holds for the averaging and splitting sets
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 minTreesPerGroup = 1,
                 OOBhonest = TRUE)

  rf <- make_savable(rf)

  spl_idx1 <- rf@R_forest[[1]]$splittingSampleIndex
  avg_idx1 <- rf@R_forest[[1]]$averagingSampleIndex
  expect_equal(all(!((101:150) %in% spl_idx1)), TRUE)
  expect_equal(all(!((101:150) %in% avg_idx1)), TRUE)

  expect_equal(length(intersect(spl_idx1,avg_idx1)), 0)

  spl_idx2 <- rf@R_forest[[2]]$splittingSampleIndex
  avg_idx2 <- rf@R_forest[[2]]$averagingSampleIndex
  expect_equal(all(!((51:100) %in% spl_idx2)), TRUE)
  expect_equal(all(!((51:100) %in% avg_idx2)), TRUE)

  expect_equal(length(intersect(spl_idx2,avg_idx2)), 0)

  spl_idx3 <- rf@R_forest[[3]]$splittingSampleIndex
  avg_idx3 <- rf@R_forest[[3]]$averagingSampleIndex
  expect_equal(all(!((1:50) %in% spl_idx3)), TRUE)
  expect_equal(all(!((1:50) %in% avg_idx3)), TRUE)

  expect_equal(length(intersect(spl_idx3,avg_idx3)), 0)

})
