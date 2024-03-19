#pragma once
#include "hydra_llm/clip_types.h"

namespace hydra::llm {

struct ClusteringWorkspace {
  std::map<size_t, const Eigen::VectorXd*> features;
  std::map<size_t, NodeId> node_lookup;
  std::map<NodeId, size_t> order;
  std::map<EdgeKey, double> edges;
  std::vector<size_t> assignments;

  ClusteringWorkspace(const SceneGraphLayer& layer,
                      const NodeEmbeddingMap& node_embeddings);

  ClusteringWorkspace(const SceneGraphLayer& layer,
                      const std::vector<NodeId>& nodes);

  size_t size() const;

  size_t featureDim() const;

  std::list<EdgeKey> addMerge(EdgeKey to_merge);

  std::vector<std::vector<NodeId>> getClusters() const;
};

}  // namespace hydra::llm
