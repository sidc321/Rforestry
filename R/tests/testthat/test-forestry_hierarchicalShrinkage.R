test_that("Tests hierarchical shrinkage works as expected", {
  context("Check total node number equals length of full weights and count vector")

  #set.seed(238943202)
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

  context("Check output predictions when using hierarchical shrinkage for small tree")

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
  shrinked_pred = predict(rf, x_test, hier_shrinkage=T, lambda_shrinkage=lambda_shrinkage)
  
  expect_equal(shrinked_pred,expectedPredictions)

  context("Check output predictions when using hierarchical shrinkage for lambda=0 and large lambda")
  test_idx <- sample(nrow(iris), 100)
  x_train <- data.frame(width= iris[-test_idx, -1])
  y_train <- iris[-test_idx, 1]
  x_test <- data.frame(width= iris[test_idx, -1])
  y_test <- iris[test_idx, 1]

  rf <- forestry(x = x_train, y = y_train, ntree = 10,replace=F,sampsize = length(y_train),mtry=ncol(x))
  noshrink_predictions = predict(rf,x_test)
  lambda0_predictions = predict(rf, x_test, hier_shrinkage=T, lambda_shrinkage=0)
  expect_equal(noshrink_predictions,lambda0_predictions)

  lambdalarge_predictions = predict(rf, x_test, hier_shrinkage=T, lambda_shrinkage=1e10)
  tot_prediction_diffs = lambdalarge_predictions-mean(y_train)
  expect_true(all.equal(tot_prediction_diffs ,rep(0,length(tot_prediction_diffs) )))

  context("Check hierarchical shrinkage prediction matches getOOBpreds")
  rf <- forestry(x = iris[,-1],
                 y = iris[,1],
                 OOBhonest = TRUE,ntree=10)

  doubleOOBpreds <- getOOBpreds(rf, doubleOOB = TRUE,
                                noWarning = TRUE,hier_shrinkage = T,lambda_shrinkage = 10)
  OOBpreds <- getOOBpreds(rf, noWarning = TRUE,hier_shrinkage = T,lambda_shrinkage = 10)
  predict_doubleOOBpreds <- predict(rf, aggregation = "doubleOOB",hier_shrinkage = T,lambda_shrinkage = 10)
  predict_OOBpreds <- predict(rf, aggregation = "oob",hier_shrinkage = T,lambda_shrinkage = 10)

  # Expect OOB preds from getOOB preds and predict to be the same
  expect_equal(all.equal(predict_OOBpreds,
                         OOBpreds), TRUE)

  # Expect double OOB preds to be the same from predict and getOOBpreds
  expect_equal(all.equal(predict_doubleOOBpreds,
                         doubleOOBpreds), TRUE)

  context("Check Weight matrix matches predictions from hierarchical shrinkage")
  x = iris[,-1]
  y = iris[,1]
  rf <- forestry(x=x,y=y,ntree=10)
  shrink_preds = predict(rf,x,weightMatrix = TRUE,hier_shrinkage = T,lambda_shrinkage =10)
  noshrink  = predict(rf,x,weightMatrix = TRUE)
  # now we reconstruct predictions from the weight matrix and check they match
  weight_preds = as.vector(shrink_preds$weightMatrix %*% y)
  expect_equal(all.equal(weight_preds,shrink_preds$predictions),TRUE)
  
  context("Check hierarchical shrinkage predictions and weightMatrix oob and double oob matches expectations for lambda = 0 and large lambda")

  x = iris[,-1]
  y = iris[,1]
  rf <- forestry(x,
                 y,
                 OOBhonest = TRUE)
  predict_doubleOOBpreds_lambda0 <- predict(rf, aggregation = "doubleOOB",hier_shrinkage = T,lambda_shrinkage = 0, weightMatrix = TRUE)
  predict_doubleOOBpreds <- predict(rf, aggregation = "doubleOOB", weightMatrix = TRUE)
  
  predict_OOBpreds_lambda0 <- predict(rf, aggregation = "oob",hier_shrinkage = T,lambda_shrinkage = 0, weightMatrix = TRUE)
  predict_OOBpreds <- predict(rf, aggregation = "oob", weightMatrix = TRUE)
  
  expect_equal(all.equal(predict_doubleOOBpreds_lambda0$predictions,
                         predict_doubleOOBpreds$predictions), TRUE)
  expect_equal(all.equal(predict_OOBpreds_lambda0$predictions,
                         predict_OOBpreds$predictions), TRUE)
  expect_equal(all.equal(predict_doubleOOBpreds_lambda0$weightMatrix,
                         predict_doubleOOBpreds$weightMatrix), TRUE)
  expect_equal(all.equal(predict_OOBpreds_lambda0$weightMatrix,
                         predict_OOBpreds$weightMatrix), TRUE)
  

  predict_doubleOOBpreds_lambdalarge <- predict(rf, aggregation = "doubleOOB",hier_shrinkage = T,lambda_shrinkage = 1e10)
  predict_OOBpreds_lambdalarge <- predict(rf, aggregation = "oob",hier_shrinkage = T,lambda_shrinkage = 1e10)
  
  expect_true(mean(predict_doubleOOBpreds_lambdalarge<5.92) && mean(predict_doubleOOBpreds_lambdalarge>5.75))
  expect_true(mean(predict_OOBpreds_lambdalarge<5.92) && mean(predict_OOBpreds_lambdalarge>5.75))
  
  context("Check Weight matrix for large lambda")
  x = iris[,-1]
  y = iris[,1]
  rf <- forestry(x,
                 y, sampsize=length(y), replace=FALSE,ntree=10)
  predict_matrix_lambdalarge <- predict(rf, x, hier_shrinkage = T,lambda_shrinkage = 1e10, weightMatrix = TRUE)
  expect_equal(rowSums(predict_matrix_lambdalarge$weightMatrix),rep(1,ncol(predict_matrix_lambdalarge$weightMatrix)))
  dims = dim(predict_matrix_lambdalarge$weightMatrix)
  expect_equal(all.equal(predict_matrix_lambdalarge$weightMatrix,
                         matrix(rep(1/nrow(x),prod(dims)),nrow=dims[1])
                         ),TRUE)

})
