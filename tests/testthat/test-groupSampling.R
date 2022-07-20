test_that("Tests sampling with groups", {
  context('Test leaving out groups on a small dataset ')


  x <- iris[,-1]
  y <- iris[,1]

  # Use the species as the group
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 ntree = 500,
                 minTreesPerGroup = 1)

  rf <- make_savable(rf)

  # Test that a tree has been grown leaving out each species
  # Note that because we sort by seed after training the forest
  # so the first trees in R_forest have left out the last group etc
  idx1 <- rf@R_forest[[498]]$splittingSampleIndex
  expect_equal(all(!((101:150) %in% idx1)), TRUE)

  idx2 <- rf@R_forest[[499]]$splittingSampleIndex
  expect_equal(all(!((51:100) %in% idx2)), TRUE)

  idx3 <- rf@R_forest[[500]]$splittingSampleIndex
  expect_equal(all(!((1:50) %in% idx3)), TRUE)


  # Use the species as the group + 10 trees
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 ntree = 30,
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
                 ntree=3,
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

  context("Test that ntree parameter is specified correctly")

  # Test that the forest ntree parameter is equal to max(minTreesPerGroup * |groups|, ntree)
  rf <- forestry(x = x,
                 y = y,
                 ntree = 100,
                 minTreesPerGroup = 10,
                 groups = iris$Species)

  expect_equal(rf@ntree, 100)

  rf <- forestry(x = x,
                 y = y,
                 ntree = 10,
                 minTreesPerGroup = 10,
                 groups = iris$Species)

  expect_equal(rf@ntree, 30)

  # Test note on the number of trees
  expect_output(
    rf <- forestry(x = x,
                   y = y,
                   ntree = 10,
                   seed = 83,
                   minTreesPerGroup = 1000,
                   groups = iris$Species),
    "Using 3 groups with 1000 trees per group will train 3000 trees in the forest."
  )

  context("Save and load with minTreePerGroup > ntree")
  wd <- tempdir()

  rf <- forestry(x = x,
                 y = y,
                 ntree = 10,
                 seed = 83,
                 minTreesPerGroup = 10,
                 groups = iris$Species)


  y_pred_before <- predict(rf, x, aggregation = "oob")

  saveForestry(rf, filename = file.path(wd, "forest.Rda"))
  rm(rf)
  forest_after <- loadForestry(file.path(wd, "forest.Rda"))

  y_pred_after <- predict(forest_after, x, aggregation = "oob")
  testthat::expect_equal(all.equal(y_pred_before, y_pred_after),TRUE)

  file.remove(file.path(wd, "forest.Rda"))


})
