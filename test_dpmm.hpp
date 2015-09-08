/*
 * Test DPMM by replicating what previous studies report.
 */
// Copyright (c) 2015 Takao Noguchi (tkngch@runbox.com)

#include <iostream>
#include "dpmm.hpp"

/*
 * Functions to relicate the study by Anderson (1991).
 *
 * As explained by Sanborn, Griffiths, and Navarro (2010), the exact
 * implementations used in the original study is unclear for Figure 2.
 * Currently, the test executable does not trigger replication of Figure 2.
 *
 * Anderson, J. R. (1991). The adaptive nature of human categorization.
 * Psychological Review, 98, 409-429.
 */
void replicate_Anderson(void);

void replicate_Anderson_Figure_1(void);
bool test_Anderson_Figure_1(const unsigned int index,
                            std::vector<unsigned int> stimulus,
                            DPMM::Model &model,
                            const std::vector<double> correct_values,
                            const double tolerance);

void replicate_Anderson_Figure_2(void);
bool test_Anderson_Figure_2(const double c, const double correlation,
                            const std::vector<double> estimation,
                            const std::map<std::string, double> &tolerance);

/*
 * Functions to replicate some of the results from the study by Sanborn,
 * Griffiths and Navarro (2010).
 *
 * Sanborn, A. N., Griffiths, T. L., & Navarro, D. J. (2010). Rational
 * approximations to rational models: Alternative algorithms for category
 * learning. Psychological Review, 117, 1144-1167.
 */
void replicate_SGN(void);

void replicate_SGN_Figure_8(void);
bool test_SGN_Figure_8(const DPMM::Approximation_Method approx_method,
                       const unsigned int n_particles,
                       const unsigned int n_training_blocks,
                       const std::vector<double> correlations,
                       const double tolerance);

void replicate_SGN_Figure_11(void);
bool test_SGN_Figure_11(const double coupling_parameter,
                        const std::vector<double> beta,
                        const DPMM::Approximation_Method approx_method,
                        const double correct_ssd, const double tolerance);

/*
 * This function simulates transfer learning: It trains the model first for
 * specified number of blocks and then draws predictions for testing stimuli
 * set. The model is not trained during the testing phase. The results are
 * stored in "predictions" vector.
 */
void train_model_then_predict(
    const unsigned int n_simulations, const DPMM::Parameter &parameter,
    const unsigned int n_training_blocks,
    const std::vector<std::vector<unsigned int>> &training_objects,
    const std::vector<std::vector<unsigned int>> &testing_objects,
    const std::vector<unsigned int> &training_dimensions,
    const std::vector<unsigned int> &testing_dimensions,
    const unsigned int dimension_to_be_predicted,
    std::vector<double> &predictions);

/*
 * This function simulates trial-and-error learning: It draws a prediction and
 * then present the model with a correct answer. The results are stored in
 * "predictions" vector.
 */
void predict_and_train_model(
    const unsigned int n_simulations, const DPMM::Parameter &parameter,
    const unsigned int n_blocks,
    std::vector<std::vector<unsigned int>> &objects,
    const std::vector<unsigned int> &training_dimensions,
    const std::vector<unsigned int> &testing_dimensions,
    const unsigned int dimension_to_be_predicted,
    std::vector<double> &predictions);

/*
 * Fills the vectors with the stimuli used by Medin and Schaffer (1978).
 *
 * Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification
 * learning. Psychological Review, 85, 207–238.
 */
void fill_MS_stimuli(std::vector<std::vector<unsigned int>> &training_objects,
                     std::vector<unsigned int> &training_dimensions,
                     std::vector<std::vector<unsigned int>> &testing_objects,
                     std::vector<unsigned int> &testing_dimensions,
                     unsigned int &dimension_to_be_predicted,
                     std::vector<double> &data);

/*
 * Fills the vectors with the stimuli used by Shepard, Hovland, and Jenkins
 * (1961).
 *
 * Shepard, R. N., Hovland, C. I., and Jenkins, H. M. (1961). Learning and
 * memorization of classifications. Psychological Monographs: General and
 * Applied, 75 (13).
 *
 * Coding is based on Nosofsky, R. M. (1984). Choice, similarity, and the
 * context theory of classification. Journal of Experimental Psychology:
 * Learning, Memory and Cognition, 10, 104-114.
 *
 * colour: black = 0, white = 1
 * shape: triangle = 0, square = 1
 * size: large = 0, small = 1
 */
void fill_SHJ_stimuli(const unsigned int type_id,
                      std::vector<std::vector<unsigned int>> &objects,
                      std::vector<unsigned int> &training_dimensions,
                      std::vector<unsigned int> &testing_dimensions,
                      unsigned int &dimension_to_be_predicted);

/*
 * Fills "data" vector with the data from the following study:
 *
 * Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., and
 * Glaughthier, P. (1994). Comparing models of rule-based classification
 * learning: A replication and extenstion of Shepard, Hovland, and Jenkins
 * (1961).
 */
void fill_NGPM_data(const unsigned int type_id, std::vector<double> &data);

/*
 * Computes Pearson correlation coefficient
 */
double cor(std::vector<double> x, std::vector<double> y);
