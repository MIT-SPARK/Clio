#pragma once
#include <hydra/common/dsg_types.h>

#include <Eigen/Dense>
#include <memory>
#include <set>

namespace hydra::llm {

struct Cluster {
  using Ptr = std::shared_ptr<Cluster>;
  std::set<NodeId> nodes;
  double score;
  Eigen::VectorXd feature;
  size_t best_task_index;
  std::string best_task_name;
};

}  // namespace hydra::llm
