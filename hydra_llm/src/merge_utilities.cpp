#include "hydra_llm/merge_utilities.h"

namespace hydra::llm {

Eigen::VectorXd MeanMerger::merge(const Eigen::VectorXd& lhs,
                                  double,
                                  const Eigen::VectorXd& rhs,
                                  double) const {
  return (lhs + rhs) / 2.0;
}

Eigen::VectorXd WeightedMeanMerger::merge(const Eigen::VectorXd& lhs,
                                          double lhs_score,
                                          const Eigen::VectorXd& rhs,
                                          double rhs_score) const {
  return (lhs_score * lhs + rhs_score * rhs) / (lhs_score + rhs_score);
}

Eigen::VectorXd MaxMerger::merge(const Eigen::VectorXd& lhs,
                                 double lhs_score,
                                 const Eigen::VectorXd& rhs,
                                 double rhs_score) const {
  return lhs_score >= rhs_score ? lhs : rhs;
}

}  // namespace hydra::llm
