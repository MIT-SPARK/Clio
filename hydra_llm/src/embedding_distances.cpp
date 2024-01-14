#include "hydra_llm/embedding_distances.h"

namespace hydra::llm {

double CosineDistance::dist(const Eigen::VectorXd& lhs,
                            const Eigen::VectorXd& rhs) const {
  return lhs.dot(rhs);
}

double CosineDistance::score(const Eigen::VectorXd& lhs,
                             const Eigen::VectorXd& rhs) const {
  // map [-1, 1] to [0, 1]
  return 0.5f * dist(lhs, rhs) + 0.5f;
}

double L2Norm::dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs - rhs).norm();
}

double L2Norm::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // map [2, 0] to [0, 1]
  return 1.0 - 0.5 * dist(lhs, rhs);
}

double L1Norm::dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs - rhs).lpNorm<1>();
}

double L1Norm::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // map [2, 0] to [0, 1]
  return 1.0 - 0.5 * dist(lhs, rhs);
}

LerfScore::LerfScore(const Config& config) {
  cannonical_ = config.cannonical_features.create();
}

double LerfScore::dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // technically the softmax isn't a dist
  return 1.0 - score(lhs, rhs);
}

double LerfScore::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  const CosineDistance dot_prod;
  const auto l_dot_r = std::exp(dot_prod.dist(lhs, rhs));
  const auto cannonical_diff = cannonical_->getDistances(dot_prod, lhs).array().exp();
  return (l_dot_r / (cannonical_diff + l_dot_r)).minCoeff();
}

void declare_config(LerfScore::Config& config) {
  using namespace config;
  name("LerfScore::Config");
  field(config.cannonical_features, "cannonical_features");
}

}  // namespace hydra::llm
