#pragma once
#include <hydra/common/dsg_types.h>
#include <hydra/reconstruction/sensor.h>

#include <Eigen/Geometry>
#include <memory>

namespace hydra::llm {

struct ClipEmbedding {
  using Ptr = std::unique_ptr<ClipEmbedding>;

  explicit ClipEmbedding(const Eigen::VectorXd& vec) : embedding(vec) {}

  explicit ClipEmbedding(const std::vector<double>& vec)
      : embedding(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size())) {}

  const Eigen::VectorXd embedding;
};

struct ScoredEmbedding {
  double score;
  ClipEmbedding::Ptr clip;
  size_t task_index;
};

using NodeEmbeddingMap = std::map<NodeId, const ClipEmbedding*>;

}  // namespace hydra::llm
