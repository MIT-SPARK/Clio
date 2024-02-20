#pragma once
#include <config_utilities/factory.h>

#include <Eigen/Dense>

#include "hydra_llm/edge_selector.h"

namespace hydra::llm {

class IBEdgeSelector : public EdgeSelector {
 public:
  struct Config {
    double score_threshold = 0.25;
    double min_beta = 1.0e3;
    double tolerance = -1.0e-18;
  };

  explicit IBEdgeSelector(const Config& config);

  virtual ~IBEdgeSelector() = default;

  void setup(const ClusteringWorkspace& ws, const ScoreFunc& score_func) override;

  double scoreEdge(const ClusteringWorkspace& ws,
                   const ScoreFunc& score_func,
                   EdgeKey edge) override;

  bool updateFromEdge(const ClusteringWorkspace& ws,
                      const ScoreFunc& score_func,
                      EdgeKey edge) override;

  bool compareEdges(const std::pair<EdgeKey, double>& lhs,
                    const std::pair<EdgeKey, double>& rhs) const override;

  const Config config;

  std::string summarize() const override;

 protected:
  // p(x), p(z), p(y)
  Eigen::VectorXd px_;
  Eigen::VectorXd pz_;
  Eigen::VectorXd py_;
  // p(z|x), p(y|x), p(y|z)
  Eigen::MatrixXd pz_x_;  // NxN
  Eigen::MatrixXd py_x_;  // 2xN
  Eigen::MatrixXd py_z_;  // 2xN
  // mutual information caches
  double I_xz_prev_;
  double I_zy_prev_;
  std::vector<double> betas_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EdgeSelector,
                                     IBEdgeSelector,
                                     IBEdgeSelector::Config>("IB");
};

void declare_config(IBEdgeSelector::Config& config);

}  // namespace hydra::llm
