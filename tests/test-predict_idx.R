test_that("Tests predict index option", {
  context('Tests getting specific predictions for one observation')

  x <- iris[1:40,-c(1,5)]
  y <- iris[1:40,1]

  rf <- forestry(x = x,
                 y = y,
                 seed = 131,
                 OOBhonest = TRUE,
                 maxDepth = 2,
                 verbose = TRUE,
                 scale = FALSE,
                 ntree = 3)

  rf <- make_savable(rf)
  idx <- 4
  insample_idx <- sort(union(rf@R_forest[[idx]]$averagingSampleIndex,
                             rf@R_forest[[idx]]$splittingSampleIndex))

  p_out <- predict(rf, newdata = x, predictIdx = c(1), weightMatrix = TRUE)
  p_std <- predict(rf, newdata = x, weightMatrix = TRUE)
  p_oob <- predict(rf, aggregation = "oob")

})
