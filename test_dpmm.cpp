// Copyright (c) 2015 Takao Noguchi (tkngch@runbox.com)

#include "test_dpmm.hpp"

int main(void) {

  replicate_Anderson();
  replicate_SGN();

  return 0;
}

void replicate_Anderson(void) {
  replicate_Anderson_Figure_1();
  replicate_Anderson_Figure_2();
}

void replicate_Anderson_Figure_1(void) {
  const double tolerance = 0.0001;
  bool pass = true;

  const DPMM::Parameter parameter{
      0.5, {1, 1, 1, 1, 1}, DPMM::Approximation_Method::local_map};
  const std::vector<unsigned int> n_uniques = {2, 2, 2, 2, 2};
  DPMM::Model model{parameter, n_uniques};

  pass *= test_Anderson_Figure_1(0, std::vector<unsigned int>{1, 1, 1, 1, 1},
                                 model, std::vector<double>{0.0313}, tolerance);

  pass *=
      test_Anderson_Figure_1(1, std::vector<unsigned int>{1, 0, 1, 0, 1}, model,
                             std::vector<double>{0.0165, 0.0157}, tolerance);

  pass *=
      test_Anderson_Figure_1(2, std::vector<unsigned int>{1, 0, 1, 1, 0}, model,
                             std::vector<double>{0.0235, 0.0104}, tolerance);

  pass *=
      test_Anderson_Figure_1(3, std::vector<unsigned int>{0, 0, 0, 0, 0}, model,
                             std::vector<double>{0.0029, 0.0078}, tolerance);

  pass *= test_Anderson_Figure_1(
      4, std::vector<unsigned int>{0, 1, 0, 1, 1}, model,
      std::vector<double>{0.0035, 0.0033, 0.0063}, tolerance);

  pass *= test_Anderson_Figure_1(
      5, std::vector<unsigned int>{0, 1, 0, 0, 0}, model,
      std::vector<double>{0.0013, 0.0110, 0.0054, 0.0052}, tolerance);

  if (pass) {
    std::cout << "Anderson Figure 1 Replication: Pass." << std::endl;
  }
}

bool test_Anderson_Figure_1(const unsigned int index,
                            std::vector<unsigned int> stimulus,
                            DPMM::Model &model,
                            const std::vector<double> correct_values,
                            const double tolerance) {
  bool pass = true;
  const std::vector<unsigned int> visible_dimensions = {0, 1, 2, 3, 4};

  model.learn(stimulus, visible_dimensions);

  for (unsigned int i = 0; i < correct_values.size(); i++) {
    if (fabs(model.particles.at(0).cluster_weights.at(i) -
             correct_values.at(i)) > tolerance) {
      std::cout << "Anderson Figure 1 Replication: Failed at Level " << index
                << ". "
                << "What should be " << correct_values.at(i) << " is "
                << model.particles.at(0).cluster_weights.at(i) << "."
                << std::endl;
      pass = false;
    }
  }
  return pass;
}

void replicate_Anderson_Figure_2(void) {
  bool pass = true;

  std::map<std::string, double> tolerance;
  tolerance["estimation"] = 0.1;
  tolerance["correlation"] = 0.1;

  /* These values are obtained through eye-balling, and are most likely to be
   * incorrect.
   */
  pass *= test_Anderson_Figure_2(
      0.25, 0.87, std::vector<double>{0.55, 0.58, 0.58, 0.58, 0.58, 0.49, 0.58,
                                      0.48, 0.48, 0.49, 0.46, 0.45},
      tolerance);

  pass *= test_Anderson_Figure_2(
      0.50, 0.66, std::vector<double>{0.59, 0.48, 0.54, 0.55, 0.55, 0.45, 0.55,
                                      0.46, 0.46, 0.55, 0.40, 0.36},
      tolerance);

  pass *= test_Anderson_Figure_2(
      0.75, 0.43, std::vector<double>{0.51, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
                                      0.50, 0.50, 0.51, 0.49, 0.48},
      tolerance);

  if (pass) {
    std::cout << "Anderson Figure 2 Replication: Pass." << std::endl;
  }
}

bool test_Anderson_Figure_2(const double c, const double correlation,
                            const std::vector<double> estimation,
                            const std::map<std::string, double> &tolerance) {
  bool pass = true;

  std::vector<std::vector<unsigned int>> training_objects;
  std::vector<std::vector<unsigned int>> testing_objects;
  std::vector<unsigned int> training_dimensions;
  std::vector<unsigned int> testing_dimensions;
  unsigned int dimension_to_be_predicted;
  std::vector<double> data(12);
  fill_MS_stimuli(training_objects, training_dimensions, testing_objects,
                  testing_dimensions, dimension_to_be_predicted, data);

  const unsigned int n_simulations = 1000;
  const unsigned int n_training_blocks = 1;

  const DPMM::Parameter parameter{c, std::vector<double>{1, 1, 1, 1, 1},
                                  DPMM::Approximation_Method::local_map};
  std::vector<double> predictions;

  train_model_then_predict(n_simulations, parameter, n_training_blocks,
                           training_objects, testing_objects,
                           training_dimensions, testing_dimensions,
                           dimension_to_be_predicted, predictions);

  for (unsigned int j = 0; j < estimation.size(); j++) {
    if (fabs(predictions.at(j) - estimation.at(j)) >
        tolerance.at("estimation")) {
      std::cout << "Anderson Figure 2 Replication: Failed at c = " << c << ". "
                << "Estimated probability should be " << estimation.at(j)
                << ", "
                << "but the model returned " << predictions.at(j) << "."
                << std::endl;
      pass = false;
    }
  }
  const double r = cor(predictions, data);
  if (fabs(r - correlation) > tolerance.at("correlation")) {
    std::cout << "Anderson Figure 2 Replication: Failed at c = " << c << ". "
              << "Correlation should be " << correlation << ", "
              << "but the model returned " << r << "." << std::endl;
    pass = false;
  }

  return pass;
}

void replicate_SGN(void) {
  replicate_SGN_Figure_8();
  replicate_SGN_Figure_11();
}

void replicate_SGN_Figure_8(void) {
  bool pass = true;

  // With 1000 simulations, the estimate is unreliable and depends on random
  // seeds. Set generous tolerance here.
  const double tolerance = 0.02;

  /* top row */
  int n_training_blocks = 1;
  pass *= test_SGN_Figure_8(DPMM::Approximation_Method::local_map, 1,
                            n_training_blocks, std::vector<double>{0.88, 0.87},
                            tolerance);
  pass *= test_SGN_Figure_8(DPMM::Approximation_Method::particle_filter, 1,
                            n_training_blocks, std::vector<double>{0.84, 0.77},
                            tolerance);
  pass *= test_SGN_Figure_8(DPMM::Approximation_Method::particle_filter, 100,
                            n_training_blocks, std::vector<double>{0.84, 0.78},
                            tolerance);

  /* bottom row */
  n_training_blocks = 10;
  pass *= test_SGN_Figure_8(DPMM::Approximation_Method::local_map, 1,
                            n_training_blocks, std::vector<double>{0.87, 0.95},
                            tolerance);
  pass *= test_SGN_Figure_8(DPMM::Approximation_Method::particle_filter, 1,
                            n_training_blocks, std::vector<double>{0.90, 0.87},
                            tolerance);
  pass *= test_SGN_Figure_8(DPMM::Approximation_Method::particle_filter, 100,
                            n_training_blocks, std::vector<double>{0.93, 0.90},
                            tolerance);

  if (pass) {
    std::cout << "SGN Figure 8 Replication: Pass." << std::endl;
  }
}

bool test_SGN_Figure_8(const DPMM::Approximation_Method approx_method,
                       const unsigned int n_particles,
                       const unsigned int n_training_blocks,
                       const std::vector<double> correlations,
                       const double tolerance) {
  bool pass = true;

  std::vector<std::vector<unsigned int>> training_objects;
  std::vector<std::vector<unsigned int>> testing_objects;
  std::vector<unsigned int> training_dimensions;
  std::vector<unsigned int> testing_dimensions;
  unsigned int dimension_to_be_predicted;
  std::vector<double> data(12);
  fill_MS_stimuli(training_objects, training_dimensions, testing_objects,
                  testing_dimensions, dimension_to_be_predicted, data);

  const unsigned int n_simulations = 1000;
  const std::vector<double> coupling_parameters{0.1, 0.3};

  const std::vector<unsigned int> n_uniques = {2, 2, 2, 2, 2};
  const std::vector<double> beta = {1, 1, 1, 1, 1};

  std::vector<double> predictions;

  for (int i = 0; i < 2; i++) {
    DPMM::Parameter parameter{coupling_parameters.at(i), beta, approx_method,
                              n_particles};

    train_model_then_predict(n_simulations, parameter, n_training_blocks,
                             training_objects, testing_objects,
                             training_dimensions, testing_dimensions,
                             dimension_to_be_predicted, predictions);

    const double r = cor(predictions, data);
    if (fabs(r - correlations.at(i)) > tolerance) {
      std::cout << "SGN Figure 8 Replication: Failed. "
                << parameter.get_approximation_method_name()
                << " with c = " << coupling_parameters.at(i) << ". "
                << "Correlation should be " << correlations.at(i) << ", "
                << "but the model returned " << r << "." << std::endl;
      pass = false;
    }
  }
  return pass;
}

void replicate_SGN_Figure_11(void) {
  bool pass = true;

  // With 1000 simulations, the estimate is unreliable and depends on random
  // seeds. Set generous tolerance here.
  const double tolerance = 0.02;

  // local map
  pass *= test_SGN_Figure_11(0.3, std::vector<double>{0.5, 0.5, 0.5, 0.01},
                             DPMM::Approximation_Method::local_map, 0.31,
                             tolerance);

  // particle filter
  pass *= test_SGN_Figure_11(0.3, std::vector<double>{0.1, 0.1, 0.1, 0.1},
                             DPMM::Approximation_Method::particle_filter, 0.24,
                             tolerance);

  if (pass) {
    std::cout << "SGN Figure 11 Replication: Pass." << std::endl;
  }
}

bool test_SGN_Figure_11(const double coupling_parameter,
                        const std::vector<double> beta,
                        const DPMM::Approximation_Method approx_method,
                        const double correct_ssd, const double tolerance) {
  DPMM::Parameter parameter{coupling_parameter, beta, approx_method, 1};

  bool pass = true;

  std::vector<std::vector<unsigned int>> objects;
  std::vector<std::vector<unsigned int>> objects_;
  std::vector<unsigned int> training_dimensions;
  std::vector<unsigned int> testing_dimensions;
  unsigned int dimension_to_be_predicted;

  std::vector<double> data;
  std::vector<unsigned int> structure;
  std::vector<unsigned int> structure_copy;

  const unsigned int n_blocks = 16;
  const unsigned int n_simulations = 100;

  std::vector<double> predictions(n_blocks);
  double ssd = 0; // sum squared deviation

  for (unsigned int type_id = 1; type_id < 7; type_id++) {
    fill_SHJ_stimuli(type_id, objects, training_dimensions, testing_dimensions,
                     dimension_to_be_predicted);
    fill_SHJ_stimuli(type_id, objects_, training_dimensions, testing_dimensions,
                     dimension_to_be_predicted);
    // each block has 16 trials: each stimulus twice.
    objects.insert(end(objects), begin(objects_), end(objects_));

    fill_NGPM_data(type_id, data);

    predict_and_train_model(n_simulations, parameter, n_blocks, objects,
                            training_dimensions, testing_dimensions,
                            dimension_to_be_predicted, predictions);

    for (unsigned int i = 0; i < n_blocks; i++) {
      ssd +=
          (predictions.at(i) - data.at(i)) * (predictions.at(i) - data.at(i));
    }
  }

  if (fabs(ssd - correct_ssd) > tolerance) {
    std::cout << "SGN Figure 11 Replication: Failed. "
              << "SSD for " << parameter.get_approximation_method_name()
              << " should be " << correct_ssd << ", but it is " << ssd << "."
              << std::endl;
    pass = false;
  }

  return pass;
}

void train_model_then_predict(
    const unsigned int n_simulations, const DPMM::Parameter &parameter,
    const unsigned int n_training_blocks,
    const std::vector<std::vector<unsigned int>> &training_objects,
    const std::vector<std::vector<unsigned int>> &testing_objects,
    const std::vector<unsigned int> &training_dimensions,
    const std::vector<unsigned int> &testing_dimensions,
    const unsigned int dimension_to_be_predicted,
    std::vector<double> &predictions) {

  std::vector<std::vector<unsigned int>> all_objects = training_objects;
  all_objects.insert(end(all_objects), begin(testing_objects),
                     end(testing_objects));

  std::vector<unsigned int> n_uniques;
  std::vector<std::set<unsigned int>> unique_values;
  DPMM::Misc::get_unique_values(all_objects, unique_values, n_uniques);

  DPMM::Model model{parameter, n_uniques};

  std::vector<std::vector<double>> accuracy;
  accuracy.resize(n_simulations);
  for (unsigned int sim = 0; sim < n_simulations; sim++) {
    model.forget();
    for (unsigned int block = 0; block < n_training_blocks; block++) {
      model.learn(training_objects, training_dimensions);
    }

    model.predict(testing_objects, testing_dimensions,
                  dimension_to_be_predicted, accuracy.at(sim));
  }

  predictions.resize(testing_objects.size());
  std::fill(begin(predictions), end(predictions), 0.0);

  std::for_each(begin(accuracy), end(accuracy),
                [&predictions](const std::vector<double> &vec) {
                  const unsigned int n = vec.size();
                  for (unsigned int i = 0; i < n; i++) {
                    predictions.at(i) += vec.at(i);
                  }
                });

  std::transform(begin(predictions), end(predictions), begin(predictions),
                 [&](double x) { return x / n_simulations; });
}

void predict_and_train_model(
    const unsigned int n_simulations, const DPMM::Parameter &parameter,
    const unsigned int n_blocks,
    std::vector<std::vector<unsigned int>> &objects,
    const std::vector<unsigned int> &training_dimensions,
    const std::vector<unsigned int> &testing_dimensions,
    const unsigned int dimension_to_be_predicted,
    std::vector<double> &predictions) {

  std::vector<unsigned int> n_uniques;
  std::vector<std::set<unsigned int>> unique_values;
  DPMM::Misc::get_unique_values(objects, unique_values, n_uniques);

  DPMM::Model model{parameter, n_uniques};

  const unsigned int n_objects = objects.size();
  std::vector<double> accuracy;
  accuracy.resize(n_objects);

  std::mt19937 random_generator;
  const unsigned int seed = 1234567890;
  random_generator.seed(seed);

  for (unsigned int sim = 0; sim < n_simulations; sim++) {
    for (unsigned int block = 0; block < n_blocks; block++) {
      std::shuffle(begin(objects), end(objects), random_generator);

      for (unsigned int i = 0; i < n_objects; i++) {
        model.predict(objects.at(i), testing_dimensions,
                      dimension_to_be_predicted, accuracy.at(i));
        model.learn(objects.at(i), training_dimensions);
      }
      predictions.at(block) +=
          std::accumulate(begin(accuracy), end(accuracy), 0.0) / n_objects;
    }
    model.forget();
  }

  std::transform(begin(predictions), end(predictions), begin(predictions),
                 [=](double x) { return 1.0 - (x / n_simulations); });
}

void fill_MS_stimuli(std::vector<std::vector<unsigned int>> &training_objects,
                     std::vector<unsigned int> &training_dimensions,
                     std::vector<std::vector<unsigned int>> &testing_objects,
                     std::vector<unsigned int> &testing_dimensions,
                     unsigned int &dimension_to_be_predicted,
                     std::vector<double> &data) {
  training_objects.assign({{1, 1, 1, 1, 1},
                           {1, 0, 1, 0, 1},
                           {1, 0, 1, 1, 0},
                           {0, 0, 0, 0, 0},
                           {0, 1, 0, 1, 1},
                           {0, 1, 0, 0, 0}});
  training_dimensions.assign({0, 1, 2, 3, 4});

  testing_objects.assign({{1, 1, 1, 1, 1},
                          {0, 1, 0, 1, 1},
                          {1, 0, 1, 0, 1},
                          {1, 1, 0, 1, 1},
                          {0, 1, 1, 1, 1},
                          {0, 0, 0, 1, 1},
                          {1, 1, 1, 0, 1},
                          {1, 0, 0, 0, 1},
                          {0, 0, 1, 0, 1},
                          {1, 0, 1, 1, 1},
                          {0, 1, 0, 0, 1},
                          {0, 0, 0, 0, 1}});
  testing_dimensions.assign({0, 1, 2, 3});
  dimension_to_be_predicted = 4;

  data.assign({4.8, 4.8, 4.6, 4.4, 4.3, 3.8, 3.6, 3.5, 3.0, 2.5, 2.1, 1.8});
}

void fill_SHJ_stimuli(const unsigned int type_id,
                      std::vector<std::vector<unsigned int>> &objects,
                      std::vector<unsigned int> &training_dimensions,
                      std::vector<unsigned int> &testing_dimensions,
                      unsigned int &dimension_to_be_predicted) {
  // color (white = 1), size (small = 1), shape (square = 1)

  objects.clear();
  training_dimensions.clear();
  testing_dimensions.clear();

  switch (type_id) {
  case 1: // Type I
    objects.assign({{0, 0, 0, 0},
                    {0, 0, 1, 0},
                    {0, 1, 0, 0},
                    {0, 1, 1, 0},
                    {1, 0, 0, 1},
                    {1, 0, 1, 1},
                    {1, 1, 0, 1},
                    {1, 1, 1, 1}});
    break;
  case 2: // Type II
    objects.assign({{0, 0, 0, 0},
                    {0, 0, 1, 1},
                    {0, 1, 0, 0},
                    {0, 1, 1, 1},
                    {1, 0, 0, 1},
                    {1, 0, 1, 0},
                    {1, 1, 0, 1},
                    {1, 1, 1, 0}});
    break;
  case 3: // Type III
    objects.assign({{0, 0, 0, 0},
                    {0, 0, 1, 0},
                    {0, 1, 0, 0},
                    {0, 1, 1, 1},
                    {1, 0, 0, 1},
                    {1, 0, 1, 1},
                    {1, 1, 0, 0},
                    {1, 1, 1, 1}});
    break;
  case 4: // Type IV
    objects.assign({{0, 0, 0, 0},
                    {0, 0, 1, 0},
                    {0, 1, 0, 0},
                    {0, 1, 1, 1},
                    {1, 0, 0, 0},
                    {1, 0, 1, 1},
                    {1, 1, 0, 1},
                    {1, 1, 1, 1}});
    break;
  case 5: // Type V
    objects.assign({{0, 0, 0, 0},
                    {0, 0, 1, 0},
                    {0, 1, 0, 0},
                    {0, 1, 1, 1},
                    {1, 0, 0, 1},
                    {1, 0, 1, 1},
                    {1, 1, 0, 1},
                    {1, 1, 1, 0}});
    break;
  case 6: // Type VI
    objects.assign({{0, 0, 0, 0},
                    {0, 0, 1, 1},
                    {0, 1, 0, 1},
                    {0, 1, 1, 0},
                    {1, 0, 0, 1},
                    {1, 0, 1, 0},
                    {1, 1, 0, 0},
                    {1, 1, 1, 1}});
    break;
  }

  training_dimensions.assign({0, 1, 2, 3});
  testing_dimensions.assign({0, 1, 2});
  dimension_to_be_predicted = 3;
}

void fill_NGPM_data(const unsigned int type_id, std::vector<double> &data) {

  data.reserve(25);
  switch (type_id) {
  case 1:
    data.assign({0.211, 0.025, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000});
    break;
  case 2:
    data.assign({0.378, 0.156, 0.083, 0.056, 0.031, 0.027, 0.028, 0.016, 0.016,
                 0.008, 0.000, 0.002, 0.005, 0.003, 0.002, 0.000, 0.000, 0.000,
                 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000});
    break;
  case 3:
    data.assign({0.459, 0.286, 0.223, 0.145, 0.081, 0.078, 0.063, 0.033, 0.023,
                 0.016, 0.019, 0.009, 0.008, 0.013, 0.009, 0.013, 0.008, 0.006,
                 0.009, 0.003, 0.005, 0.000, 0.003, 0.005, 0.002});
    break;
  case 4:
    data.assign({0.422, 0.295, 0.222, 0.172, 0.148, 0.109, 0.089, 0.063, 0.025,
                 0.031, 0.019, 0.025, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000,
                 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000});
    break;
  case 5:
    data.assign({0.472, 0.331, 0.230, 0.139, 0.106, 0.081, 0.067, 0.078, 0.048,
                 0.045, 0.050, 0.036, 0.031, 0.027, 0.016, 0.014, 0.014, 0.014,
                 0.013, 0.014, 0.013, 0.009, 0.011, 0.008, 0.008});
    break;
  case 6:
    data.assign({0.498, 0.341, 0.284, 0.245, 0.217, 0.192, 0.192, 0.177, 0.172,
                 0.128, 0.139, 0.117, 0.103, 0.098, 0.106, 0.106, 0.078, 0.077,
                 0.078, 0.061, 0.058, 0.042, 0.042, 0.030, 0.038});
    break;
  default:
    std::cout << "Invalid type_id: " << type_id << std::endl;
    throw;
  }
}

double cor(std::vector<double> x, std::vector<double> y) {
  assert(x.size() == y.size());

  const int n = x.size();
  double numerator = 0;

  auto sum = [](std::vector<double> x) {
    return std::accumulate(begin(x), end(x), 0.0);
  };

  const double x_mean = sum(x) / n;
  std::transform(begin(x), end(x), begin(x),
                 [=](double d) { return d - x_mean; });
  const double y_mean = sum(y) / n;
  std::transform(begin(y), end(y), begin(y),
                 [=](double d) { return d - y_mean; });

  for (int i = 0; i < n; i++) {
    numerator += x.at(i) * y.at(i);
  }

  std::vector<double> x2(x.size());
  std::transform(begin(x), end(x), begin(x2),
                 [](double d) { return std::pow(d, 2.0); });
  std::vector<double> y2(y.size());
  std::transform(begin(y), end(y), begin(y2),
                 [](double d) { return std::pow(d, 2.0); });

  const double denominator = std::sqrt(sum(x2)) * std::sqrt(sum(y2));

  const double r = numerator / denominator;
  return r;
}
