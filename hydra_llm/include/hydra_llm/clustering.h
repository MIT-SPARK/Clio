#pragma once
#include <config_utilities/virtual_config.h>

#include <Eigen/Dense>

#include "hydra_llm/clip_types.h"
#include "hydra_llm/cluster.h"
#include "hydra_llm/embedding_group.h"

namespace hydra::llm {

class Clustering {
 public:
  using Clusters = std::vector<Cluster::Ptr>;
  struct Config {
    config::VirtualConfig<EmbeddingGroup> tasks;
  };

  Clustering(const Config& config);

  virtual ~Clustering() = default;

  virtual Clusters cluster(const SceneGraphLayer& layer,
                           const NodeEmbeddingMap& embeddings) const = 0;

  const Config config;

 protected:
  EmbeddingGroup::Ptr tasks_;
};

void declare_config(Clustering::Config& config);

}  // namespace hydra::llm
