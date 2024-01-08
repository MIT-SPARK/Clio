#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/dsg_types.h>

#include <Eigen/Dense>

#include "hydra_llm/embedding_norms.h"
#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

class Clustering {
  struct Config {
    config::VirtualConfig<EmbeddingNorm> norm;
    double stop_value = 0.0;
  };

  Clustering(const Config& config);

  const Config config;

 private:
  std::unique_ptr<EmbeddingNorm> norm_;

 public:
  const EmbeddingNorm& norm;
  TaskEmbeddings::Ptr tasks;

 protected:
  Eigen::MatrixXd scoreMatrices(const std::vector<Eigen::VectorXd>& assignments);

  std::vector<double> computePhi(const SceneGraphLayer& layer,
                                 const Eigen::MatrixXd& scores);

  void cluster(const SceneGraphLayer& layer, const std::vector<Eigen::VectorXd>& nodes);
};

}  // namespace hydra::llm
