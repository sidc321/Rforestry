test_that("Tests symmetry in multiple features", {

  library(Rforestry)
  context("1-dimensional example")

  set.seed(23322)
  n <- 100
  x <- matrix(runif(n,min=-2,max=2), ncol=1)
  y <- x[,1]**3
  colnames(x) <- c("V1")
  # colnames(x) <- c("V1","V2")
  # plot(x[,1],y)
  # x[135:235,1] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=1,
                 seed=212342,
                 maxDepth = 2,
                 #mtry=2,
                 #OOBhonest = TRUE,
                 scale = FALSE,
                 #monotonicConstraints = c(1),
                 #monotoneAvg = TRUE,
                 symmetric = c(1)
                 )

  p <- predict(rf, newdata = x)
  # plot(x[,1],p)

  context("2-dimensional example")

  set.seed(23322)
  n <- 1000
  x <- matrix(runif(2*n,min=-2,max=2), ncol=2)
  y <- x[,2]**3
  colnames(x) <- c("V1","V2")
  # plot(x[,1],y)
  # x[135:235,1] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=212342,
                 #maxDepth = 3,
                 #mtry=2,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(0,1),
                 symmetric = c(0,1))

  p <- predict(rf, newdata = x)
  # plot(x[,2],p)

  context("2-dimensional example in second feature")

  set.seed(23322)
  n <- 1000
  x <- matrix(runif(2*n,min=-2,max=2), ncol=2)
  y <- x[,1]**3
  colnames(x) <- c("V1","V2")
  # plot(x[,2],y)
  # x[135:235,1] <- NA

  rf <- forestry(x=x,
                 y=y,
                 ntree=500,
                 seed=212342,
                 #maxDepth = 3,
                 #mtry=2,
                 OOBhonest = TRUE,
                 scale = FALSE,
                 monotonicConstraints = c(1,0),
                 monotoneAvg = TRUE,
                 symmetric = c(1,0))

  p <- predict(rf, newdata = x)
  # plot(x[,1],p)

})
