#pragma once

#include <spark_dsg/dynamic_scene_graph.h>

#include <Eigen/Dense>

#include "hydra_llm/clustering_workspace.h"
#include "hydra_llm/embedding_distances.h"
#include "hydra_llm/embedding_group.h"
namespace hydra::llm {

struct PyGivenXConfig {
  double score_threshold = 0.23;
  size_t top_k = 2;
  bool cumulative = true;
  bool null_task_preprune = true;
};

Eigen::MatrixXd computeIBpyGivenX(const ClusteringWorkspace& ws,
                                  const EmbeddingGroup& tasks,
                                  const EmbeddingDistance& metric,
                                  const PyGivenXConfig& config);

Eigen::VectorXd computeIBpx(const ClusteringWorkspace& ws);

Eigen::VectorXd computeIBpy(const EmbeddingGroup& tasks);

double computeDeltaWeight(const SceneGraphLayer& layer,
                          const std::vector<NodeId>& nodes);
}  // namespace hydra::llm
