[![R-CMD-check](https://github.com/forestry-labs/Rforestry/actions/workflows/check-noncontainerized.yaml/badge.svg)](https://github.com/forestry-labs/Rforestry/actions/workflows/check-noncontainerized.yaml)

## Rforestry: Random Forests, Linear Trees, and Gradient Boosting for Inference and Interpretability

Sören Künzel, Theo Saarinen, Simon Walter, Sam Antonyan, Edward Liu, Allen Tang, Jasjeet Sekhon


## Introduction

Rforestry is a fast implementation of Honest Random Forests, Gradient Boosting,
and Linear Random Forests, with an emphasis on inference and interpretability.

## How to install
1. The GFortran compiler has to be up to date. GFortran Binaries can be found [here](https://gcc.gnu.org/wiki/GFortranBinaries).
2. The [devtools](https://github.com/r-lib/devtools) package has to be installed. You can install it using,  `install.packages("devtools")`.
3. The package contains compiled code, and you must have a development environment to install the development version. You can use `devtools::has_devel()` to check whether you do. If no development environment exists, Windows users download and install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and macOS users download and install [Xcode](https://apps.apple.com/us/app/xcode/id497799835).
4. The latest development version can then be installed using
`devtools::install_github("forestry-labs/Rforestry")`. For Windows users, you'll need to skip 64-bit compilation `devtools::install_github("forestry-labs/Rforestry", INSTALL_opts = c('--no-multiarch'))` due to an outstanding gcc issue.


## Python Package Usage

```R
library(Rforestry)




set.seed(292315)
test_idx <- sample(nrow(iris), 3)
x_train <- iris[-test_idx, -1]
y_train <- iris[-test_idx, 1]
x_test <- iris[test_idx, -1]

rf <- forestry(x = x_train, y = y_train, nthread = 2)

predict(rf, x_test)
```

## R Package Usage

```R
library(Rforestry)




set.seed(292315)
test_idx <- sample(nrow(iris), 3)
x_train <- iris[-test_idx, -1]
y_train <- iris[-test_idx, 1]
x_test <- iris[test_idx, -1]

rf <- forestry(x = x_train, y = y_train, nthread = 2)

predict(rf, x_test)
```



