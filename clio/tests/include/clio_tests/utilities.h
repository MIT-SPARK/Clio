#pragma once

#include <hydra/openset/embedding_distances.h>

namespace clio::test {

struct TestEmbeddingGroup : public EmbeddingGroup {
  struct Config {
    size_t num_embeddings = 1;
  };

  explicit TestEmbeddingGroup(const Config&);

  static config::VirtualConfig<EmbeddingGroup> getDefault(size_t num_embeddings = 1);

  static Eigen::MatrixXd getEmbedding(size_t index);

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingGroup, TestEmbeddingGroup, Config>(
          "test_group");
};

void declare_config(TestEmbeddingGroup::Config&);

}  // namespace clio::test
