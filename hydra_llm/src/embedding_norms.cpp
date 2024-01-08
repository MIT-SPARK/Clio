#include "hydra_llm/embedding_norms.h"

namespace hydra::llm {

double CosineDistance::norm(const Eigen::VectorXd& lhs,
                            const Eigen::VectorXd& rhs) const {
  const auto divisor = lhs.norm() * rhs.norm();
  return lhs.dot(rhs) / divisor;
}

double L2Norm::norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs - rhs).norm();
}

double L1Norm::norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs - rhs).lpNorm<1>();
}

}  // namespace hydra::llm
