#include "hydra_llm/probability_utilities.h"
#include <glog/logging.h>

namespace hydra::llm {

// -sum(p(x)log(p(x))) over all x
double shannonEntropy(const Eigen::VectorXd& p, double tolerance) {
  double entropy = 0.0;
  for (int i = 0; i < p.rows(); ++i) {
    if (p(i) < tolerance) {
      continue;
    }

    const auto p_i = p(i);
    entropy += p_i * std::log2(p_i);
  }
  return -entropy;
}

// compute the Jensen-Shannon divergence of p(a|b) (uses prior p(b))
double jensenShannonDivergence(const Eigen::MatrixXd& pa_b,
                               const Eigen::VectorXd& p_b,
                               double tolerance) {
  const Eigen::MatrixXd p_joint = pa_b * p_b;
  double total_entropy = 0.0;
  for (int i = 0; i < pa_b.rows(); ++i) {
    total_entropy += p_b(i) * shannonEntropy(pa_b.row(i), tolerance);
  }

  return shannonEntropy(p_joint, tolerance) - total_entropy;
}

// compute the mutual information between two distributions
double mutualInformation(const Eigen::VectorXd& pa,
                         const Eigen::VectorXd& pb,
                         const Eigen::MatrixXd pa_b,
                         double tolerance) {
  CHECK_EQ(pa_b.cols(), pb.rows());
  CHECK_EQ(pa_b.rows(), pa.rows());
  double total = 0.0;
  for (int b = 0; b < pa_b.cols(); ++b) {
    for (int a = 0; a < pa_b.rows(); ++a) {
      const auto p_joint = pa_b(a, b);
      // avoid log blowing up for events that can't occur
      if (p_joint < tolerance || pa(a) < tolerance) {
        continue;
      }

      total += pb(b) * p_joint * std::log2(p_joint / pa(a));
    }
  }

  return total;
}

}  // namespace hydra::llm
