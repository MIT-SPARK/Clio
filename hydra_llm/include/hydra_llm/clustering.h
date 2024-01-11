#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/dsg_types.h>

#include <Eigen/Dense>

#include "hydra_llm/clip_types.h"
#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

using NodeEmbeddingMap = std::map<NodeId, const ClipEmbedding*>;

struct Cluster {
  using Ptr = std::shared_ptr<Cluster>;
  std::set<NodeId> nodes;
  double score;
  ClipEmbedding::Ptr clip;
  size_t best_task_index;
  std::string best_task_name;
};

class Clustering {
 public:
  struct Config {
    config::VirtualConfig<TaskEmbeddings> tasks;
  };

  Clustering(const Config& config);

  virtual ~Clustering() = default;

  virtual std::vector<Cluster::Ptr> cluster(
      const SceneGraphLayer& layer, const NodeEmbeddingMap& embeddings) const = 0;

  const Config config;

 protected:
  TaskEmbeddings::Ptr tasks_;
};

void declare_config(Clustering::Config& config);

}  // namespace hydra::llm
