test_that("Tests custom sampling parameters", {


  context("Try indices in the wrong range for the customSplitSample")

  x <- iris[, -1]
  y <- iris[, 1]
  splittingSample = list(1:5, 6:10, 146:151)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 41:45)

  expect_error(
    rf <- forestry(x = x,
                 y = y,
                 customSplitSample = splittingSample,
                 customAvgSample = averagingSample,
                 customExcludeSample = excludedSample,
                 ntree = 3),
    "customSplitSample must contain positive integers up to the number of observations in x"
  )


  context("Test non integer entries to excludedSample")
  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 2.3,2.5,9.0, 41:45)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplitSample = splittingSample,
                   customAvgSample = averagingSample,
                   customExcludeSample = excludedSample,
                   ntree = 3),
    "customExcludeSample must be equal in length to ntree"
  )


  context("Test that splitting, averaging and excluded sets are disjoint")
  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 30:45)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplitSample = splittingSample,
                   customAvgSample = averagingSample,
                   customExcludeSample = excludedSample,
                   ntree = 3),
    "Excluded samples must be disjoint from averaging samples"
  )

  splittingSample = list(1:5, 6:10, 26:30)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 31:45)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplitSample = splittingSample,
                   customAvgSample = averagingSample,
                   customExcludeSample = excludedSample,
                   ntree = 3),
    "Splitting and averaging samples must be disjoint"
  )

  context("Check warning when combined with other honesty parameters")

  expect_warning(
    rf <- forestry(x = x,
                   y = y,
                   customSplitSample = list(1:10),
                   customAvgSample = list(11:20),
                   customExcludeSample = list(21:30),
                   OOBhonest = TRUE,
                   ntree = 1),
    "When customSplitSample is set, other honesty implementations are ignored"
  )

  expect_warning(
    rf <- forestry(x = x,
                   y = y,
                   customSplitSample = list(1:10),
                   customAvgSample = list(11:20),
                   customExcludeSample = list(21:30),
                   splitratio = .5,
                   ntree = 1),
    "When customSplitSample is set, other honesty implementations are ignored"
  )




  context('Tests that splitting, averaging, and excluded samples are set correctly')
  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 41:45)

  rf <- forestry(x = x,
                 y = y,
                 customSplitSample = splittingSample,
                 customAvgSample = averagingSample,
                 customExcludeSample = excludedSample,
                 ntree = 3)
  rf <- make_savable(rf)

  # Trees in the forest are currently stored sorted by seed in descending order
  # So when checking indices tree i gets (ntree - i + 1) samples
  # This is kind of messy so we probably want to change it

  for (i in rf@ntree) {
    expect_equal(rf@R_forest[[i]]$splittingSampleIndex,
                 splittingSample[[4-i]])
    expect_equal(rf@R_forest[[i]]$averagingSampleIndex,
                 averagingSample[[4-i]])
    expect_equal(rf@R_forest[[i]]$excludedSampleIndex,
                 excludedSample[[4-i]])
  }


  context("Try sampling outside of the forest construction")
  set.seed(13123)
  splittingSample = list()
  averagingSample = list()
  excludedSample= list()


  for (count in 1:10) {
    sampled_splitting <- sample(1:65, replace = TRUE, size = 65)
    sampled_avging <- sample(66:125, replace = TRUE, size = 60)
    sampled_exclude <- sample(126:150, replace = TRUE, size = 25)
    splittingSample[[count]] = sampled_splitting
    averagingSample[[count]] = sampled_avging
    excludedSample[[count]] = sampled_exclude
  }

  rf <- forestry(x = x,
                 y = y,
                 customSplitSample = splittingSample,
                 customAvgSample = averagingSample,
                 customExcludeSample = excludedSample,
                 ntree = 10)
  rf <- make_savable(rf)

  for (i in rf@ntree) {
    expect_equal(rf@R_forest[[i]]$splittingSampleIndex,
                 splittingSample[[11-i]])
    expect_equal(rf@R_forest[[i]]$averagingSampleIndex,
                 averagingSample[[11-i]])
    expect_equal(rf@R_forest[[i]]$excludedSampleIndex,
                 excludedSample[[11-i]])
  }


  context("Test the excluded set is working as expected with predict")

  rf <- forestry(x = x,
                 y = y,
                 customSplitSample = list(1:10),
                 customAvgSample = list(11:20),
                 customExcludeSample = list(21:30),
                 ntree = 1)

  p <- predict(rf, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(all.equal(p$treeCounts[1:10], rep(1,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[11:20], rep(0,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[21:30], rep(0,10)), TRUE)



  p <- predict(rf, newdata = x, aggregation = "doubleOOB", weightMatrix = TRUE)
  expect_equal(all.equal(p$treeCounts[1:10], rep(0,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[11:20], rep(0,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[21:30], rep(0,10)), TRUE)



  context("Test when we provide an empty excluded sample for a tree")
  rf <- forestry(x = x,
                 y = y,
                 customSplitSample = list(1:10,11:20),
                 customAvgSample = list(11:20,21:30),
                 customExcludeSample = list(21:30, c()),
                 ntree = 2)

  p <- predict(rf, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(all.equal(p$treeCounts[1:10], rep(2,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[11:20], rep(1,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[21:30], rep(0,10)), TRUE)



  p <- predict(rf, newdata = x, aggregation = "doubleOOB", weightMatrix = TRUE)
  expect_equal(all.equal(p$treeCounts[1:10], rep(1,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[11:20], rep(0,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[21:30], rep(0,10)), TRUE)

  rf <- make_savable(rf)
  expect_equal(length(rf@R_forest[[1]]$excludedSampleIndex), 0)
  expect_equal(length(rf@R_forest[[2]]$excludedSampleIndex), 10)
  expect_equal(all.equal(rf@R_forest[[2]]$excludedSampleIndex, c(21:30)), TRUE)

  context("Test when we no trees excluded samples")
  rf <- forestry(x = x,
                 y = y,
                 customSplitSample = list(1:10,11:20),
                 customAvgSample = list(11:20,21:30),
                 ntree = 2)

  p <- predict(rf, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(all.equal(p$treeCounts[1:10], rep(2,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[11:20], rep(1,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[21:30], rep(1,10)), TRUE)



  p <- predict(rf, newdata = x, aggregation = "doubleOOB", weightMatrix = TRUE)
  expect_equal(all.equal(p$treeCounts[1:10], rep(1,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[11:20], rep(0,10)), TRUE)
  expect_equal(all.equal(p$treeCounts[21:30], rep(1,10)), TRUE)


  rf <- make_savable(rf)

})
