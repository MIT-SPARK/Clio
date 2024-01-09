#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/dsg_types.h>

#include <Eigen/Dense>

#include "hydra_llm/embedding_norms.h"
#include "hydra_llm/merge_utilities.h"
#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

struct EdgeEmbeddingInfo {
  double weight;
  ClipEmbedding::Ptr clip;
};

using NodeEmbeddingMap = std::map<NodeId, const ClipEmbedding*>;
using EdgeEmbeddingMap = std::map<EdgeKey, EdgeEmbeddingInfo>;

class Clustering {
 public:
  struct Config {
    config::VirtualConfig<TaskEmbeddings> tasks;
    config::VirtualConfig<EmbeddingNorm> norm;
    config::VirtualConfig<EmbeddingMerger> merge;
    double stop_value = 0.0;
    double min_distance = 0.022;
  };

  Clustering(const Config& config);

  const Config config;

  void cluster(const SceneGraphLayer& layer,
               const NodeEmbeddingMap& node_embeddings) const;

 private:
  std::unique_ptr<EmbeddingNorm> norm_;
  EmbeddingMerger::Ptr embedding_merge_;
  TaskEmbeddings::Ptr tasks_;

 public:
  const EmbeddingNorm& norm;

 protected:
  void fillSubgraph(const SceneGraphLayer& layer,
                    const NodeEmbeddingMap& node_embeddings,
                    IsolatedSceneGraphLayer& cluster_layer) const;

  void computePhi(const SceneGraphLayer& layer,
                  const std::set<EdgeKey>& edges,
                  EdgeEmbeddingMap& edge_embeddings) const;
};

void declare_config(Clustering::Config& config);

}  // namespace hydra::llm