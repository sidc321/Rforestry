library(testthat)
test_that("Tests using observation weights + OOBhonesty", {
  x <- iris[,-1]
  y <- iris[,1]
  
  
  context("See if number of unique observations in each tree matches the expected value")
  rf <- forestry(x = x,
                 y = y,
                 OOBhonest = TRUE,
                 seed= 1,
                 observationWeights = c(rep(c(1),50),rep(c(1.01),50),rep(c(1.02),50))
                 )
  
  rf <- make_savable(rf)
  # Get the perdentage of unique observations in each tree
  # Overall, we expect this to be (1-((.865*n-1)/(.865*n))^n) * .865 = 0.5939828 unique
  # observations when we take a bootstrap sample of size n from the .865 unique observations
  # in the splitting and averaging set
  res = c()
  for (i in 1:500) {
    pct_unique = (length(unique(rf@R_forest[[i]]$averagingSampleIndex)) + 
                    length(unique(rf@R_forest[[i]]$splittingSampleIndex))) / 150
    res <- c(res, pct_unique)
  }
  expect_equal(mean(res), 0.5939828, tolerance = 1e-3)
  
  context("Try weights with more skew")
  # If we use weights, the expected number of unique observations goes down
  rf <- forestry(x = x,
                 y = y,
                 OOBhonest = TRUE,
                 seed= 1,
                 observationWeights = c(rep(c(1),50),rep(c(2),50),rep(c(3),50))
  )
  
  rf <- make_savable(rf)
  res = c()
  for (i in 1:500) {
    pct_unique = (length(unique(rf@R_forest[[i]]$averagingSampleIndex)) + 
                    length(unique(rf@R_forest[[i]]$splittingSampleIndex))) / 150
    res <- c(res, pct_unique)
  }
  expect_lte(mean(res), 0.5939828)
  
  
  context("See weights in each tree")
  
  res = c(0,0,0)
  for (i in 1:500) {
    num_low = length(which(union(rf@R_forest[[i]]$averagingSampleIndex, 
                                 rf@R_forest[[i]]$splittingSampleIndex) < 51))
    num_med = length(which(union(rf@R_forest[[i]]$averagingSampleIndex, 
                                 rf@R_forest[[i]]$splittingSampleIndex) < 101 & union(rf@R_forest[[i]]$averagingSampleIndex, 
                                                                                      rf@R_forest[[i]]$splittingSampleIndex) > 50))
    num_high = length(which(union(rf@R_forest[[i]]$averagingSampleIndex, 
                                 rf@R_forest[[i]]$splittingSampleIndex) > 101))
    res <- res + c(num_low,num_med,num_high)
  }
  expect_lt(res[1]/sum(res), res[2]/sum(res))
  expect_lt(res[2]/sum(res), res[3]/sum(res))

  
})
