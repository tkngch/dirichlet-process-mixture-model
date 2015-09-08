// Copyright (c) 2015 Takao Noguchi (tkngch@runbox.com)

#include "main.hpp"

int main(void) {
  test_run();

  return 0;
}

void test_run(void) {

  const std::vector<std::vector<unsigned int>> objects{
      {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
  const std::vector<unsigned int> feature_dimensions{0, 1};
  const std::vector<unsigned int> test_dimensions{2, 2, 2, 2};
  const std::vector<unsigned int> learning_dimensions{0, 1, 2};

  std::vector<unsigned int> n_uniques;
  std::vector<std::set<unsigned int>> unique_values;
  DPMM::Misc::get_unique_values(objects, unique_values, n_uniques);

  DPMM::Model model{};

  const std::vector<double> couplings{0.3, 0.3, 0.5, 1.0};
  const unsigned int n_training_blocks = 5;

  for (double coupling : couplings) {

    learn_and_predict(model, coupling, n_uniques, objects, feature_dimensions,
                      test_dimensions, learning_dimensions, n_training_blocks);
  }
}

void learn_and_predict(DPMM::Model &model, const double coupling,
                       const std::vector<unsigned int> &n_uniques,
                       const std::vector<std::vector<unsigned int>> &objects,
                       const std::vector<unsigned int> &feature_dimensions,
                       const std::vector<unsigned int> &test_dimensions,
                       const std::vector<unsigned int> &learning_dimensions,
                       const unsigned int n_training_blocks) {
  std::vector<double> accuracy;

  const DPMM::Parameter parameter{
      coupling, {1, 1, 0.1}, DPMM::Approximation_Method::particle_filter};

  model.forget();
  model.reset(parameter, n_uniques);

  for (unsigned int i = 0; i < n_training_blocks; i++) {
    model.learn(objects, learning_dimensions);
  }

  DPMM::Parameter parameter_for_prediction = parameter;
  parameter_for_prediction.c = 1.0;

  model.parameter = parameter_for_prediction;
  model.predict(objects, feature_dimensions, test_dimensions, accuracy);
  double acc =
      std::accumulate(begin(accuracy), end(accuracy), 0.0) / objects.size();

  std::cout << "coupling = " << coupling << "; mean accuracy = " << acc
            << std::endl;
  std::cout << "accuracy: [";
  std::for_each(begin(accuracy), end(accuracy),
                [](const double x) { std::cout << x << ", "; });
  std::cout << "]" << std::endl
            << std::endl;
}
