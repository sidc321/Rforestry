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
  # Get the percentage of unique observations in each tree
  # Overall, we expect this to be .632 * (1-((.632*n - 1)/(.632*n))^(.632*n)) + 
  # .233 * (1-((.233*n - 1)/(.233*n))^(.233*n)) = 0.5492571 unique
  # observations when we take a bootstrap sample of size n from the .865 unique observations
  # in the splitting and averaging set
  res = c()
  for (i in 1:500) {
    pct_unique = (length(unique(rf@R_forest[[i]]$averagingSampleIndex)) + 
                    length(unique(rf@R_forest[[i]]$splittingSampleIndex))) / 150
    res <- c(res, pct_unique)
  }
  
  expect_lt(mean(res), .632)
  skip_if_not_mac()
  expect_equal(mean(res), 0.5492571, tolerance = 1e-2)
  
  context("Try weights with more skew, expect less unique observations")
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
  expect_lte(mean(res), 0.5492571)
  
  
  context("See weights in each tree, in both sets, they should match obs weights")
  
  res_spl = c(0,0,0)
  res_avg = c(0,0,0)
  for (i in 1:500) {
    num_spl_low = length(which(rf@R_forest[[i]]$splittingSampleIndex %in% c(1:50)))
    num_avg_low = length(which(rf@R_forest[[i]]$averagingSampleIndex %in% c(1:50)))
    
    num_spl_med = length(which(rf@R_forest[[i]]$splittingSampleIndex %in% c(51:100)))
    num_avg_med = length(which(rf@R_forest[[i]]$averagingSampleIndex %in% c(51:100)))
    
    num_spl_high = length(which(rf@R_forest[[i]]$splittingSampleIndex %in% c(101:150)))
    num_avg_high = length(which(rf@R_forest[[i]]$averagingSampleIndex %in% c(101:150)))
    
    res_spl <- res_spl + c(num_spl_low,num_spl_med,num_spl_high)
    res_avg <- res_avg + c(num_avg_low,num_avg_med,num_avg_high)
  }
  
  expect_lt(res_spl[1]/sum(res_spl), res_spl[2]/sum(res_spl))
  expect_lt(res_spl[2]/sum(res_spl), res_spl[3]/sum(res_spl))
  
  expect_lt(res_avg[1]/sum(res_avg), res_avg[2]/sum(res_avg))
  expect_lt(res_avg[2]/sum(res_avg), res_avg[3]/sum(res_avg))
  
  
  skip_if_not_mac()
  expect_equal(res_spl[1]/sum(res_spl), 0.1686042, tolerance = 1e-5)
  expect_equal(res_spl[2]/sum(res_spl), 0.3330208, tolerance = 1e-5)
  expect_equal(res_spl[3]/sum(res_spl), 0.498375, tolerance = 1e-5)
  
  

  
})
