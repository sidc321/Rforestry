test_that("Tests hierarchical shrinkage works as expected", {
  context("check total node number equals length of full weights and count vector")

  set.seed(238943202)
  # Test example with simple step functio
  x <- iris[, -1]
  y <- iris[, 1]
  rf <- forestry(x,
                y,
                ntree = 10,
                maxDepth = 5)
  rf <- make_savable(rf)

  # Now check things are the right size
  var_id  = rf@R_forest[[1]]$var_id
  num_nodes = length(var_id[var_id>=0]) + length(var_id[var_id < 0])/2
  expect_equal(length(rf@R_forest[[1]]$average_count),num_nodes)
  expect_equal(length(rf@R_forest[[1]]$weightsFull),num_nodes)

  context("check output predictions when using hierarchical shrinkage for small tree")

  # Test example with simple step functio
  test_idx <- sample(nrow(iris), 100)
  x_train <- data.frame(width= iris[-test_idx, 2])
  y_train <- iris[-test_idx, 1]
  x_test <- data.frame(width= iris[test_idx, 2])

  rf <- forestry(x = x_train, y = y_train, ntree = 1, maxDepth = 1,scale=F)

  predict(rf, x_test)
  rf= make_savable(rf)
  plot(rf)
  fdata <- rf@R_forest[[1]]
  #classify training
  expectedPredictions = numeric(length(x_test[,1]))
  for(i in 1:length(expectedPredictions)){
    if(x_test[i,1]<fdata$split_val[1]){
      expectedPredictions[i] = 1
    } else{
      expectedPredictions[i]=2
    }
  }
  lambda_shrinkage=2
  weightLeftPath = fdata$weightsFull[1]*(1-1/(1+lambda_shrinkage/fdata$average_count[1])) + fdata$weightsFull[2]/(1+lambda_shrinkage/fdata$average_count[1])
  weightRightPath = fdata$weightsFull[1]*(1-1/(1+lambda_shrinkage/fdata$average_count[1])) + fdata$weightsFull[3]/(1+lambda_shrinkage/fdata$average_count[1])
  expectedPredictions[expectedPredictions==1] = weightLeftPath
  expectedPredictions[expectedPredictions==2] = weightRightPath
  shrinked_pred = predict(rf, x_test, hier_shrinkage=T, lambda_shrinkage=2)

  expect_equal(shrinked_pred,expectedPredictions)

  context("check output predictions when using hierarchical shrinkage for lambda=0")
  test_idx <- sample(nrow(iris), 100)
  x_train <- data.frame(width= iris[-test_idx, 2])
  y_train <- iris[-test_idx, 1]
  x_test <- data.frame(width= iris[test_idx, 2])

  rf <- forestry(x = x_train, y = y_train, ntree = 10)
  unshrink_predictions = predict(rf,x_test)
  lambda0_predictions = predict(rf, x_test, hier_shrinkage=T, lambda_shrinkage=0)

  context("check output predictions when using hierarchical shrinkage for large lambda")
  test_idx <- sample(nrow(iris), 100)
  x_train <- data.frame(width= iris[-test_idx, 2])
  y_train <- iris[-test_idx, 1]
  x_test <- data.frame(width= iris[test_idx, 2])

  rf <- forestry(x = x_train, y = y_train, ntree = 10)
  unshrink_predictions = predict(rf,x_test)
  lambda0_predictions = predict(rf, x_test, hier_shrinkage=T, lambda_shrinkage=1e10)
  tot_prediction_diffs = sum(abs(lambda0_predictions-lambda0_predictions[0]))
  expect_equal(0, tot_prediction_diffs)
})
