# Sourced at the top of every experiment script, with the working directory set
# to the root of the repository. It clears the workspace, compiles the C++ code
# in src/ and loads the R functions in R/.
rm(list = ls(all = TRUE))
options(show.error.locations = TRUE)

library(parallel)
library(Rcpp)
library(RcppArmadillo)
sourceCpp("src/utilitiesc.cpp")
source("R/utilitiesr.R")
source("R/smc_runner.R")
source("R/dc_abf_runner.R")

# One BLAS and OpenMP thread by default; the fusions set their own OpenMP
# thread count.
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  library(RhpcBLASctl)
  blas_set_num_threads(1)
  omp_set_num_threads(1)
}
