// Copyright (c) 2015 Takao Noguchi (tkngch@runbox.com)

#include <iostream>
#include "dpmm.hpp"

void test_run(void);

void learn_and_predict(DPMM::Model &model, const double coupling,
                       const std::vector<unsigned int> &n_uniques,
                       const std::vector<std::vector<unsigned int>> &objects,
                       const std::vector<unsigned int> &feature_dimensions,
                       const std::vector<unsigned int> &test_dimensions,
                       const std::vector<unsigned int> &learning_dimensions,
                       const unsigned int n_training_blocks);
