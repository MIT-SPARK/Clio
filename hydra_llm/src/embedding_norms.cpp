#include "hydra_llm/embedding_norms.h"

namespace hydra::llm {

double CosineDistance::norm(const Eigen::VectorXd& lhs,
                            const Eigen::VectorXd& rhs) const {
  const auto divisor = lhs.norm() * rhs.norm();
  return lhs.dot(rhs) / divisor;
}

double CosineDistance::score(const Eigen::VectorXd& lhs,
                             const Eigen::VectorXd& rhs) const {
  // map [-1, 1] to [0, 1]
  return 0.5f * norm(lhs, rhs) + 0.5f;
}

double L2Norm::norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // TODO(nathan) handle normalization better
  return (lhs.normalized() - rhs.normalized()).norm();
}

double L2Norm::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // map [2, 0] to [0, 1]
  return 1.0 - 0.5 * norm(lhs, rhs);
}

double L1Norm::norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs.normalized() - rhs.normalized()).lpNorm<1>();
}

double L1Norm::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // map [2, 0] to [0, 1]
  return 1.0 - 0.5 * norm(lhs, rhs);
}

}  // namespace hydra::llm
