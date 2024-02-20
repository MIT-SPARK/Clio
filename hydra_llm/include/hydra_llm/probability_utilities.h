#pragma once
#include <Eigen/Dense>

namespace hydra::llm {

// -sum(p(x)log(p(x))) over all x
double shannonEntropy(const Eigen::VectorXd& p, double tolerance = 1.0e-9);

// compute the Jensen-Shannon divergence of p(a|b) (uses prior p(b))
double jensenShannonDivergence(const Eigen::MatrixXd& pa_b,
                               const Eigen::VectorXd& p_b,
                               double tolerance = 1.0e-9);

// compute the mutual information between two distributions (p(a) and p(b))
double mutualInformation(const Eigen::VectorXd& pa,
                         const Eigen::VectorXd& pb,
                         const Eigen::MatrixXd pa_b,
                         double tolerance = 1.0e-9);

}  // namespace hydra::llm
