test_that("Tests custom sampling parameters", {


  context("Try indices in the wrong range for the customSplittingSample")

  x <- iris[, -1]
  y <- iris[, 1]
  splittingSample = list(1:5, 6:10, 146:151)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 41:45)

  expect_error(
    rf <- forestry(x = x,
                 y = y,
                 customSplittingSample = splittingSample,
                 customAveragingSample = averagingSample,
                 customExcludedSample = excludedSample,
                 ntree = 3),
    "customSplittingSample must contain positive integers up to the number of observations in x"
  )


  context("Test non integer entries to excludedSample")
  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 2.3,2.5,9.0, 41:45)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplittingSample = splittingSample,
                   customAveragingSample = averagingSample,
                   customExcludedSample = excludedSample,
                   ntree = 3),
    "customExcludedSample must contain positive integers up to the number of observations in x"
  )


  context("Test that splitting, averaging and excluded sets are disjoint")
  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 30:45)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplittingSample = splittingSample,
                   customAveragingSample = averagingSample,
                   customExcludedSample = excludedSample,
                   ntree = 3),
    "Excluded samples must be disjoint from splitting and averaging samples"
  )

  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 141:146)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplittingSample = splittingSample,
                   customAveragingSample = averagingSample,
                   customExcludedSample = excludedSample,
                   ntree = 3),
    "Excluded samples must be disjoint from splitting and averaging samples"
  )

  splittingSample = list(1:5, 6:10, 26:30)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 31:45)

  expect_error(
    rf <- forestry(x = x,
                   y = y,
                   customSplittingSample = splittingSample,
                   customAveragingSample = averagingSample,
                   customExcludedSample = excludedSample,
                   ntree = 3),
    "Splitting and averaging samples must be disjoint"
  )



  context('Tests that splitting, averaging, and excluded samples are set correctly')
  splittingSample = list(1:5, 6:10, 146:150)
  averagingSample = list(16:20, 21:25, 26:30)
  excludedSample = list(31:35, 36:40, 41:45)

  rf <- forestry(x = x,
                 y = y,
                 customSplittingSample = splittingSample,
                 customAveragingSample = averagingSample,
                 customExcludedSample = excludedSample,
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

})
