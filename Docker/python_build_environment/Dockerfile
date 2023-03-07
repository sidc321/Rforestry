FROM quay.io/pypa/manylinux2014_x86_64:latest

RUN yum install -y gcc-c++ clang R-core R-core-devel armadillo-devel
RUN R -e "install.packages(c('Rcpp','RcppArmadillo'), repos = 'http://cran.us.r-project.org')"