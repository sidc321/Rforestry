test_that("Tests custom sampling parameters", {
  context('Tests that splitting, averaging, and excluded samples are set correctly')

  x <- iris[, -1]
  y <- iris[, 1]
  splittingSample = list(1:5, 6:10, 146:149)
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
