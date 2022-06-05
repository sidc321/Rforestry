[![R-CMD-check](https://github.com/forestry-labs/Rforestry/actions/workflows/check-noncontainerized.yaml/badge.svg)](https://github.com/forestry-labs/Rforestry/actions/workflows/check-noncontainerized.yaml)

## Rforestry: Random Forests, Linear Trees, and Gradient Boosting for Inference and Interpretability

Sören Künzel, Theo Saarinen, Simon Walter, Sam Antonyan, Edward Liu, Allen Tang, Jasjeet Sekhon


## Introduction

Rforestry is a fast implementation of Honest Random Forests, Gradient Boosting,
and Linear Random Forests, with an emphasis on inference and interpretability.

## How to install - R Package
1. The GFortran compiler has to be up to date. GFortran Binaries can be found [here](https://gcc.gnu.org/wiki/GFortranBinaries).
2. The [devtools](https://github.com/r-lib/devtools) package has to be installed. You can install it using,  `install.packages("devtools")`.
3. The package contains compiled code, and you must have a development environment to install the development version. You can use `devtools::has_devel()` to check whether you do. If no development environment exists, Windows users download and install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and macOS users download and install [Xcode](https://apps.apple.com/us/app/xcode/id497799835).
4. The latest development version can then be installed using
`devtools::install_github("forestry-labs/Rforestry")`. For Windows users, you'll need to skip 64-bit compilation `devtools::install_github("forestry-labs/Rforestry", INSTALL_opts = c('--no-multiarch'))` due to an outstanding gcc issue.


## How to install - Python Package

The python package must be compiled before it can be used. For example, one can run:

```
mkdir build
cd build
cmake ..
make

```

## Python Package Usage

Then the python code can be called:

```
import numpy as np
import pandas as pd
import warnings
import math
import os
from random import randrange
import sys
from forestry import forestry
import Py_preprocessing
from sklearn.datasets import load_iris


data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
X = df.loc[:, df.columns != 'target']
y = df['target']

fr = forestry(ntree = 500)

print("Fitting the forest")
fr.fit(X, y)


print("Predicting with the forest")
forest_preds = fr.predict(aggregation='oob')

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
