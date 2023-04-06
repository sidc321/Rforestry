FROM quay.io/pypa/manylinux2014_x86_64:latest

RUN yum install -y gcc-c++ clang armadillo-devel
RUN curl -O https://cdn.rstudio.com/r/centos-7/pkgs/R-4.0.0-1-1.x86_64.rpm && yum install -y R-4.0.0-1-1.x86_64.rpm 

ENV PATH=/opt/rh/devtoolset-10/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/R/4.0.0/bin

# we need to run R package here for cross-language testing, so we need all dependncies for Rforestry package installed
RUN mkdir -p ~/.R && echo 'CXX17=g++ -std=gnu++17 -fPIC' > ~/.R/Makevars
RUN R -e "install.packages(c('Rcpp', 'RcppArmadillo', 'RcppThread', 'visNetwork', 'glmnet', 'onehot', 'pROC'), repos = 'http://cran.us.r-project.org')"