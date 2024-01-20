#include "hydra_llm/embedding_distances.h"

#include <glog/logging.h>

namespace hydra::llm {

double CosineDistance::dist(const Eigen::VectorXd& lhs,
                            const Eigen::VectorXd& rhs) const {
  // map [-1, 1] to [0, 2]
  return 1.0 - score(lhs, rhs);
}

double CosineDistance::score(const Eigen::VectorXd& lhs,
                             const Eigen::VectorXd& rhs) const {
  const auto divisor = std::max(lhs.norm() * rhs.norm(), config.tolerance);
  return lhs.dot(rhs) / divisor;

  return dist(lhs, rhs);
}

double L2Norm::dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs.normalized() - rhs.normalized()).norm();
}

double L2Norm::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // map [2, 0] to [0, 1]
  return 1.0 - 0.5 * dist(lhs, rhs);
}

double L1Norm::dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs.normalized() - rhs.normalized()).lpNorm<1>();
}

double L1Norm::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // map [2, 0] to [0, 1]
  return 1.0 - 0.5 * dist(lhs, rhs);
}

LerfScore::LerfScore(const Config& config) : config(config) {
  cannonical_ = config.cannonical_features.create();
}

double LerfScore::dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  // technically the softmax isn't a dist
  return 1.0 - score(lhs, rhs);
}

double LerfScore::score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  CosineDistance::Config dot_config;
  dot_config.tolerance = config.tolerance;
  const CosineDistance dot_prod(dot_config);
  // score maps to cosine similiarity (for CosineDistance norm)
  const auto l_dot_r = std::exp(dot_prod.score(lhs, rhs));
  const auto scores = cannonical_->getScores(dot_prod, lhs);
  const auto divisor = (scores.array().exp() + l_dot_r);
  return (l_dot_r / divisor).minCoeff();
}

void declare_config(LerfScore::Config& config) {
  using namespace config;
  name("LerfScore::Config");
  field(config.cannonical_features, "cannonical_features");
  field(config.tolerance, "tolerance");
}

}  // namespace hydra::llm
