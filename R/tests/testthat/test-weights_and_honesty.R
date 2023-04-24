library(testthat)
test_that("Tests using observation weights + OOBhonesty", {
  x <- iris[,-1]
  y <- iris[,1]
  
  rf <- forestry(x = x,
                 y = y,
                 OOBhonest = TRUE
                 # observationWeights = c(rep(c(1),50),rep(c(2),50),rep(c(3),50))
                 )
  
  rf <- make_savable(rf)
  pct_unique = (length(unique(rf@R_forest[[1]]$averagingSampleIndex)) + 
          length(unique(rf@R_forest[[1]]$splittingSampleIndex))) / 150
  
})
