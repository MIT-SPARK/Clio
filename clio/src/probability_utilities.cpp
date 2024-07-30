#include "clio/probability_utilities.h"

#include <glog/logging.h>
#include <hydra/utils/printing.h>

namespace clio {

// -sum(p(x)log(p(x))) over all x
double shannonEntropy(const Eigen::Ref<const Eigen::VectorXd>& p, double tolerance) {
  const auto fmt = hydra::getDefaultFormat();
  double entropy = 0.0;
  for (int i = 0; i < p.rows(); ++i) {
    const double p_i = p(i, 0);
    if (p_i < tolerance) {
      continue;
    }

    entropy += p_i * std::log2(p_i);
  }

  VLOG(30) << "p: " << p.format(fmt) << ", H: " << -entropy;
  return -entropy;
}

// compute the Jensen-Shannon divergence of p(a|b) (uses prior p(b))
double jensenShannonDivergence(const Eigen::MatrixXd& pa_b,
                               const Eigen::VectorXd& p_b,
                               double tolerance) {
  const auto fmt = hydra::getDefaultFormat();
  VLOG(30) << "=========================";
  VLOG(30) << "p(a|b): " << pa_b.format(fmt) << ", p(b): " << p_b.format(fmt);

  // M = sum_i=0^|{1, 2, ... m}| p_i(x) p(y|x=i)
  // p(a|b) * p(b) = \sum_i=0^|a| p(a|b=i)p(b=i)
  const Eigen::VectorXd M = pa_b * p_b;
  VLOG(30) << "M: " << M.format(fmt);

  VLOG(30) << "-------------------------";
  VLOG(30) << "Entropies";
  VLOG(30) << "-------------------------";

  double total_entropy = 0.0;
  for (int i = 0; i < pa_b.cols(); ++i) {
    total_entropy += p_b(i) * shannonEntropy(pa_b.col(i), tolerance);
  }
  VLOG(30) << "-------------------------";
  VLOG(30) << "=========================";
  return shannonEntropy(M, tolerance) - total_entropy;
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

}  // namespace clio
