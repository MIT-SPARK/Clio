#include "clio_tests/utilities.h"

#include <algorithm>

namespace clio::test {

TestEmbeddingGroup::TestEmbeddingGroup(const Config& config) {
  embeddings.push_back(Eigen::VectorXd::Zero(10));
  for (size_t i = 0; i < config.num_embeddings; ++i) {
    embeddings.push_back(TestEmbeddingGroup::getEmbedding(i));
  }
}

config::VirtualConfig<EmbeddingGroup> TestEmbeddingGroup::getDefault(
    size_t num_embeddings) {
  Config config;
  config.num_embeddings = num_embeddings;
  return {config, "test_group"};
}

Eigen::MatrixXd TestEmbeddingGroup::getEmbedding(size_t index) {
  Eigen::MatrixXd feature = Eigen::MatrixXd::Zero(10, 1);
  feature(std::clamp<size_t>(index, 0, 10), 0) = 1.0;
  return feature;
}

void declare_config(TestEmbeddingGroup::Config& config) {
  using namespace config;
  name("TestEmbeddingGroup::Config");
  field(config.num_embeddings, "num_embeddings");
}

}  // namespace clio::test
