#pragma once
#include <hydra/common/dsg_types.h>

#include <Eigen/Dense>
#include <map>

namespace hydra::llm {

using NodeEmbeddingMap = std::map<NodeId, const Eigen::VectorXd*>;

}  // namespace hydra::llm
