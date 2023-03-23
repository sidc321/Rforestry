test_that("Tests custom sampling parameters", {


  context("Test monotonicity with custom sampling")

  x <- data.frame(matrix(runif(100*3),ncol=3))
  y <- x[,2] + x[,1] + rnorm(100,sd = .1)

  sampling <- function(seed,
                       n=100) {
    set.seed(seed)
    splitting <- sample(1:n, replace = TRUE)
    out <- (1:n)[!(1:n %in% sort(splitting))]
    averaging <- sample(out, size = length(out), replace = TRUE)
    double_out <- out[!(out %in% sort(averaging))]
    return(list("split" = splitting, "avg" = averaging, "exclude" = double_out))
  }

  customSpl = list()
  customAvg = list()
  customEx = list()

  for (i in 1:200) {
    samp = sampling(i,n=100)
    customSpl[[i]] = samp$split
    customAvg[[i]] = samp$avg
    customEx[[i]] = samp$exclude
  }

  rf.monotone <- forestry(x = x,
                          y = y,
                          seed = 101,
                          customSplitSample = customSpl,
                          customAvgSample = customAvg,
                          customExcludeSample = customEx,
                          monotonicConstraints = c(1,0,0),
                          ntree = 200)

  new.x <- data.frame(X1 = seq(from = -2, to = 2, length.out = 100), X2 = rep(0,100), X3 = rep(0,100))

  pred.new <- predict(rf.monotone, newdata = new.x)

  # Expect monotonicity in expectation, but not strictly
  expect_gt(mean(order(pred.new)[51:100]),
            mean(order(pred.new)[1:50]))

  context("Test monotoneAvg with custom sampling")

  rf.strict <- forestry(x = x,
                        y = y,
                        seed = 100,
                        customSplitSample = customSpl,
                        customAvgSample = customAvg,
                        customExcludeSample = customEx,
                        monotonicConstraints = c(1,0,0),
                        monotoneAvg = TRUE,
                        ntree = 200)

  pred.new <- predict(rf.strict, newdata = new.x)

  # Now expect the order to be strictly increasing
  skip_if_not_mac()
  expect_equal(all.equal(order(pred.new), 1:100), TRUE)


  context("Test custom sampling with groups during aggregation")


  x = iris[,-1]
  y = iris[,1]

  customSpl = list(21:35)
  customAvg = list(1:15)

  #customEx = list(1:10,11:20,21:30,31:40)

  rf.groups <- forestry(x = x,
                        y = y,
                        seed = 100,
                        customSplitSample = customSpl,
                        customAvgSample = customAvg,
                        groups = as.factor(as.vector(sapply(1:15, function(x){return(rep(x,10))}))),
                        ntree = 1)

  context("Test OOB + doubleOOB predictions using custom sampling")
  # Cannot predict for averaging groups
  p.oob <- predict(rf.groups, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(all.equal(p.oob$treeCounts[1:20], rep(0,20)), TRUE)

  # Cannot predict for averaging or splitting groups
  p.doob <- predict(rf.groups, newdata = x, aggregation = "doubleOOB", weightMatrix = TRUE)
  expect_equal(all.equal(p.doob$treeCounts[1:40], rep(0,40)), TRUE)


  customSpl = list(21:35)
  customAvg = list(1:15)
  customEx = list(c(101,111,121,131,141))


  rf.groups.excluded <- forestry(x = x,
                                 y = y,
                                 seed = 100,
                                 customSplitSample = customSpl,
                                 customAvgSample = customAvg,
                                 customExcludeSample = customEx,
                                 groups = as.factor(as.vector(sapply(1:15, function(x){return(rep(x,10))}))),
                                 ntree = 1)


  context("Test OOB + doubleOOB predictions using custom sampling and excluded indices")
  # Now cannot predict for averaging groups or any of the excluded groups
  p.oob <- predict(rf.groups.excluded, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(all.equal(p.oob$treeCounts[c(1:20)], rep(0,20)), TRUE)
  expect_equal(all.equal(p.oob$treeCounts[c(101:150)], rep(0,50)), TRUE)


  p.doob <- predict(rf.groups.excluded, newdata = x, aggregation = "doubleOOB", weightMatrix = TRUE)
  expect_equal(all.equal(p.doob$treeCounts[c(1:40)], rep(0,40)), TRUE)
  expect_equal(all.equal(p.doob$treeCounts[c(101:150)], rep(0,50)), TRUE)



  context("Test groups + monotonicity + customSampling")

  x = iris[,-1]
  y = iris[,1]

  customSpl = list()
  customAvg = list()
  customEx = list()

  # Scheme where every group has 2 consequitive observations Group1 = 1,2  Group 2= 3,4 ...
  # Do random assignment at the group level, then flip a coin to decide which observation goes in
  for (i in 1:200) {
    samp = sampling(i,n=75)
    set.seed(i)
    customSpl[[i]] = 2*samp$split - rbinom(n=length(samp$split),size=1,prob=.5)
    customAvg[[i]] = 2*samp$avg - rbinom(n=length(samp$avg),size=1,prob=.5)
    customEx[[i]] = 2*samp$exclude - rbinom(n=length(samp$exclude),size=1,prob=.5)
  }

  rf.groups.monotone <- forestry(x = x,
                                 y = y,
                                 seed = 100,
                                 customSplitSample = customSpl,
                                 customAvgSample = customAvg,
                                 customExcludeSample = customEx,
                                 monotonicConstraints = c(1,0,0,0),
                                 groups = as.factor(as.vector(sapply(1:75, function(x){return(rep(x,2))}))),
                                 ntree = 200)


  Sepal.Length = seq(from = 2.2, to = 4.5, length.out = 10)
  Petal.Length = 1.5
  Petal.Width = .2
  Species = "setosa"

  new.x <- expand.grid(Sepal.Length,Petal.Length,Petal.Width,Species)
  names(new.x) <- names(x)

  pred.new <- predict(rf.groups.monotone, newdata = new.x)

  # Expect monotonicity in expectation, but not strictly
  expect_gt(mean(order(pred.new)[6:10]),
            mean(order(pred.new)[1:5]))

  context("Test groups + monotonicity + monotoneAvg + customSampling")

  rf.groups.monotone.strict <- forestry(x = x,
                                        y = y,
                                        seed = 100,
                                        customSplitSample = customSpl,
                                        customAvgSample = customAvg,
                                        customExcludeSample = customEx,
                                        monotonicConstraints = c(1,0,0,0),
                                        monotoneAvg = TRUE,
                                        groups = as.factor(as.vector(sapply(1:75, function(x){return(rep(x,2))}))),
                                        ntree = 200)

  pred.new <- predict(rf.groups.monotone.strict, newdata = new.x)

  # Now expect the order to be strictly increasing
  skip_if_not_mac()
  expect_equal(all.equal(order(pred.new), 1:10), TRUE)



  context("Test the groups aggregations are working with groups + monotonicity + monotoneAvg + customSampling")


  tree.groups.monotone.strict <- forestry(x = x,
                                          y = y,
                                          seed = 100,
                                          customSplitSample = list(2*(1:15)-1),
                                          customAvgSample = list(2*(16:30)-1),
                                          customExcludeSample = list(c(61,63)),
                                          monotonicConstraints = c(1,0,0,0),
                                          monotoneAvg = TRUE,
                                          groups = as.factor(as.vector(sapply(1:75, function(x){return(rep(x,2))}))),
                                          ntree = 1)


  # Now cannot predict for averaging groups or any of the excluded groups
  p.oob <- predict(tree.groups.monotone.strict, newdata = x, aggregation = "oob", weightMatrix = TRUE)
  expect_equal(all.equal(p.oob$treeCounts[c(31:60)], rep(0,30)), TRUE)
  expect_equal(all.equal(p.oob$treeCounts[c(61:64)], rep(0,4)), TRUE)


  p.doob <- predict(tree.groups.monotone.strict, newdata = x, aggregation = "doubleOOB", weightMatrix = TRUE)
  expect_equal(all.equal(p.doob$treeCounts[c(31:60)], rep(0,30)), TRUE)
  expect_equal(all.equal(p.doob$treeCounts[c(1:30)], rep(0,30)), TRUE)
  expect_equal(all.equal(p.doob$treeCounts[c(61:64)], rep(0,4)), TRUE)


  context("Test when customSamples don't belong to disjoint groups")

  expect_error(
    tree.groups.check <- forestry(x = data.frame(x1 = rnorm(12)),
                                          y = rnorm(12),
                                          seed = 100,
                                          customSplitSample = list(c(1,3,7)),
                                          customAvgSample = list(c(2,4,6)),
                                          groups = as.factor(as.vector(sapply(1:6, function(x){return(rep(x,2))}))),
                                          ntree = 1),
    "Splitting and averaging samples must contain disjoint groups"
  )

  expect_error(
    tree.groups.check <- forestry(x = data.frame(x1 = rnorm(12)),
                                  y = rnorm(12),
                                  seed = 100,
                                  customSplitSample = list(c(3,4,5,6)),
                                  customAvgSample = list(c(1,2,7)),
                                  customExcludeSample = list(c(8)),
                                  groups = as.factor(as.vector(sapply(1:6, function(x){return(rep(x,2))}))),
                                  ntree = 1),
    "Excluded samples must contain groups disjoint from those in the averaging samples"
  )


})
