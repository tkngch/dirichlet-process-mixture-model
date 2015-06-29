
#include <algorithm>
#include <iostream>
#include <Rcpp.h>
#include "dpmm.hpp"

RcppExport SEXP predict_and_learn(SEXP objects_, SEXP coupling_, SEXP beta_,
                                  SEXP n_particles_, SEXP training_dimensions_,
                                  SEXP testing_dimensions_,
                                  SEXP dimension_to_be_predicted_) {

  const double coupling = Rcpp::as<double>(coupling_);
  const std::vector<double> beta = Rcpp::as<std::vector<double>>(beta_);
  const unsigned int n_particles = Rcpp::as<unsigned int>(n_particles_);

  DPMM::Approximation_Method approx_method =
      DPMM::Approximation_Method::particle_filter;
  if (n_particles == 0) {
    approx_method = DPMM::Approximation_Method::local_map;
  }

  DPMM::Parameter parameter_for_learning{coupling, beta, approx_method,
                                         n_particles};
  DPMM::Parameter parameter_for_prediction = parameter_for_learning;
  parameter_for_prediction.c = 1.0;  // prevent a new cluster from being created

  const std::vector<std::vector<unsigned int>> objects =
      Rcpp::as<std::vector<std::vector<unsigned int>>>(objects_);

  std::vector<std::set<unsigned int>> unique_values;
  std::vector<unsigned int> n_uniques;
  DPMM::Misc::get_unique_values(objects, unique_values, n_uniques);

  const unsigned int n_objects = objects.size();

  const std::vector<unsigned int> training_dimensions =
      Rcpp::as<std::vector<unsigned int>>(training_dimensions_);

  const std::vector<unsigned int> testing_dimensions =
      Rcpp::as<std::vector<unsigned int>>(testing_dimensions_);

  const unsigned int dimension_to_be_predicted =
      Rcpp::as<unsigned int>(dimension_to_be_predicted_);

  DPMM::Model model{parameter_for_learning, n_uniques};

  std::vector<double> accuracy;
  accuracy.resize(n_objects);

  for (unsigned int i = 0; i < n_objects; i++) {
    model.parameter = parameter_for_prediction;
    model.predict(objects.at(i), testing_dimensions, dimension_to_be_predicted,
                  accuracy.at(i));

    model.parameter = parameter_for_learning;
    model.learn(objects.at(i), training_dimensions);
  }

  return Rcpp::wrap(accuracy);
}
