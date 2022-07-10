test_that("Tests predict index option", {
  context('Tests getting specific predictions for one observation')

  x <- iris[1:40,-c(1,5)]
  y <- iris[1:40,1]

  rf <- forestry(x = x,
                 y = y,
                 seed = 131,
                 OOBhonest = TRUE,
                 maxDepth = 2,
                 scale = FALSE,
                 ntree = 1)

  rf <- make_savable(rf)
  insample_idx <- sort(union(rf@R_forest[[1]]$averagingSampleIndex,
                             rf@R_forest[[1]]$splittingSampleIndex))

  p_out <- predict(rf, newdata = x, predictIdx = c(1), weightMatrix = TRUE)
  p_oob <- predict(rf, aggregation = "oob")

})
