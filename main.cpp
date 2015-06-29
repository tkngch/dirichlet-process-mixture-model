
#include "main.hpp"

int main(void) {
  test_run();

  return 0;
}

void test_run(void) {

  std::vector<std::vector<unsigned int>> objects{
      {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
  std::vector<unsigned int> feature_dimensions{0, 1};
  std::vector<unsigned int> test_dimensions{2, 2, 2, 2};
  std::vector<unsigned int> learning_dimensions{0, 1, 2};

  std::vector<unsigned int> n_uniques;
  std::vector<std::set<unsigned int>> unique_values;
  DPMM::Misc::get_unique_values(objects, unique_values, n_uniques);

  DPMM::Model model{};

  std::map<std::string, double> cluster_stats;
  std::vector<double> accuracy;

  const std::vector<double> couplings{0.3, 0.3, 0.5, 1.0};
  for (double coupling : couplings) {
    const DPMM::Parameter parameter{
        coupling, {1, 1, 0.1}, DPMM::Approximation_Method::particle_filter};

    model.forget();
    model.reset(parameter, n_uniques);

    model.learn(objects, learning_dimensions);
    model.learn(objects, learning_dimensions);
    model.learn(objects, learning_dimensions);

    DPMM::Parameter parameter_for_prediction = parameter;
    parameter_for_prediction.c = 1.0;

    model.parameter = parameter_for_prediction;
    model.predict(objects, feature_dimensions, test_dimensions, accuracy);
    double acc =
        std::accumulate(begin(accuracy), end(accuracy), 0.0) / objects.size();

    model.get_cluster_statistics(learning_dimensions, unique_values,
                                 cluster_stats);

    std::cout << "coupling = " << coupling << "; accuracy = " << acc
              << "; cluster stats. n = " << cluster_stats.at("n")
              << ", informativeness = " << cluster_stats.at("informativeness")
              << ", entropy = " << cluster_stats.at("entropy") << std::endl;
    std::cout << "accuracy: [";
    std::for_each(begin(accuracy), end(accuracy),
                  [](const double x) { std::cout << x << ", "; });
    std::cout << "]" << std::endl;
  }
}
