test_that("Tests sampling with groups", {
  context('Test leaving out groups on a small dataset ')


  x <- iris[,-1]
  y <- iris[,1]

  # Use the species as the group
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 ntree = 3,
                 minTreesPerFold = 1,
                 foldSize = 1,
                 seed =  101
                 )

  rf <- make_savable(rf)

  # Test that a tree has been grown leaving out each species
  # Note that because we sort by seed after training the forest
  # so the first trees in R_forest have left out the last group etc
  idx1 <- rf@R_forest[[1]]$splittingSampleIndex
  skip_if_not_mac()
  expect_equal(all(!((101:150) %in% idx1)), TRUE)

  idx2 <- rf@R_forest[[2]]$splittingSampleIndex
  skip_if_not_mac()
  expect_equal(all(!((51:100) %in% idx2)), TRUE)

  idx3 <- rf@R_forest[[3]]$splittingSampleIndex
  skip_if_not_mac()
  expect_equal(all(!((1:50) %in% idx3)), TRUE)


  # Use the species as the group + 10 trees
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 ntree = 30,
                 seed =  101,
                 minTreesPerFold = 10)

  rf <- make_savable(rf)


  expect_equal(length(rf@R_forest), 30)

  for ( i in 1:30) {
    idx <- rf@R_forest[[i]]$splittingSampleIndex

    # Expect all observations to fall into a single group
    expect_equal(all(!((101:150) %in% idx)) || all(!((51:100) %in% idx)) || all(!((1:50) %in% idx)), TRUE)
  }


  # Test when using honesty, this holds for the averaging and splitting sets
  rf <- forestry(x = x,
                 y = y,
                 groups = iris$Species,
                 minTreesPerFold = 1,
                 ntree=3,
                 seed =  101,
                 OOBhonest = TRUE)

  rf <- make_savable(rf)

  spl_idx1 <- rf@R_forest[[1]]$splittingSampleIndex
  avg_idx1 <- rf@R_forest[[1]]$averagingSampleIndex
  skip_if_not_mac()
  expect_equal(all(!((101:150) %in% spl_idx1)), TRUE)
  skip_if_not_mac()
  expect_equal(all(!((101:150) %in% avg_idx1)), TRUE)

  expect_equal(length(intersect(spl_idx1,avg_idx1)), 0)

  spl_idx2 <- rf@R_forest[[2]]$splittingSampleIndex
  avg_idx2 <- rf@R_forest[[2]]$averagingSampleIndex
  skip_if_not_mac()
  expect_equal(all(!((51:100) %in% spl_idx2)), TRUE)
  skip_if_not_mac()
  expect_equal(all(!((51:100) %in% avg_idx2)), TRUE)

  expect_equal(length(intersect(spl_idx2,avg_idx2)), 0)

  spl_idx3 <- rf@R_forest[[3]]$splittingSampleIndex
  avg_idx3 <- rf@R_forest[[3]]$averagingSampleIndex
  skip_if_not_mac()
  expect_equal(all(!((1:50) %in% spl_idx3)), TRUE)
  skip_if_not_mac()
  expect_equal(all(!((1:50) %in% avg_idx3)), TRUE)

  expect_equal(length(intersect(spl_idx3,avg_idx3)), 0)

  context("Test that ntree parameter is specified correctly")

  # Test that the forest ntree parameter is equal to max(minTreesPerFold * # folds, ntree)
  rf <- forestry(x = x,
                 y = y,
                 ntree = 100,
                 minTreesPerFold = 10,
                 seed =  101,
                 groups = iris$Species)

  expect_equal(rf@ntree, 100)

  rf <- forestry(x = x,
                 y = y,
                 ntree = 10,
                 seed =  101,
                 minTreesPerFold = 10,
                 groups = iris$Species)

  expect_equal(rf@ntree, 30)

  # Test note on the number of trees
  expect_output(
    rf <- forestry(x = x,
                   y = y,
                   ntree = 10,
                   seed = 83,
                   minTreesPerFold = 1000,
                   groups = iris$Species),
    "Using 3 folds with 1000 trees per group will train 3000 trees in the forest."
  )

  context("Save and load with minTreePerGroup > ntree")
  wd <- tempdir()

  rf <- forestry(x = x,
                 y = y,
                 ntree = 10,
                 seed = 83,
                 minTreesPerFold = 10,
                 groups = iris$Species)


  y_pred_before <- predict(rf, x, aggregation = "oob")

  saveForestry(rf, filename = file.path(wd, "forest.Rda"))
  rm(rf)
  forest_after <- loadForestry(file.path(wd, "forest.Rda"))

  y_pred_after <- predict(forest_after, x, aggregation = "oob")
  testthat::expect_equal(all.equal(y_pred_before, y_pred_after),TRUE)

  file.remove(file.path(wd, "forest.Rda"))


  context("Test group sampling with observation weights")

  x <- iris[,-1]
  y <- iris[,1]

  # Helper function for getting empirical observation weights across trained RF
  get_weights <- function(object) {
    total_p_1 <- 0
    total_p_2 <- 0
    total_p_3 <- 0
    for (tree_i in 1:object@ntree) {
      obs_i <- object@R_forest[[tree_i]]$splittingSampleIndex

      p_1 <- (length(which(obs_i %in% 1:50)) / 150)
      p_2 <- (length(which(obs_i %in% 51:100)) / 150)
      p_3 <- (length(which(obs_i %in% 101:150)) / 150)
      total_p_1 <- total_p_1 + p_1
      total_p_2 <- total_p_2 + p_2
      total_p_3 <- total_p_3 + p_3
    }
    return(list(p1 = total_p_1 / (object@ntree - object@minTreesPerFold),
                p2 = total_p_2 / (object@ntree - object@minTreesPerFold),
                p3 = total_p_3 / (object@ntree - object@minTreesPerFold)))
  }

  forest <- forestry(x = x, y = y,
                     observationWeights = c(rep(2,50),rep(3,50),rep(5,50)),
                     minTreesPerFold = 1,
                     foldSize = 1,
                     seed = 12312,
                     groups = as.factor(iris$Species),
                     ntree= 300)
  forest <- make_savable(forest)


  w <- get_weights(forest)
  expect_lt(w$p1, .5)
  expect_lt(w$p2, .5)
  expect_gt(w$p3, .1)

  # Run exact test of proportions on Mac
  skip_if_not_mac()
  expect_equal(all.equal(unname(unlist(w)), c(0.201716833891,0.300178372352,0.501449275362), tolerance = 1e-6), TRUE)

})
