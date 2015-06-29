#!/usr/bin/env Rscript

library(Rcpp)
dyn.load("~/wip/dirichlet-process-mixture-model/libdpmm_r.so")

objects <- list(c(0, 0, 0), c(0, 1, 1), c(1, 0, 1), c(1, 1, 0))
coupling <- 0.3
beta <- c(1, 1, 0.1)
n_particles <- 1
training_dimensions <- c(0, 1, 2)
testing_dimensions <- c(0, 1)
dimension_to_be_predicted <- 2

accuracy <- .Call("predict_and_learn", objects, coupling, beta, n_particles,
                  training_dimensions, testing_dimensions, dimension_to_be_predicted)
print(accuracy)
print(mean(accuracy))
