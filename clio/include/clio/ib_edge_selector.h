#pragma once
#include <config_utilities/factory.h>

#include <Eigen/Dense>

#include "clio/edge_selector.h"
#include "clio/ib_utils.h"

namespace clio {

class IBEdgeSelector : public EdgeSelector {
 public:
  struct Config {
    double max_delta = 1.0e-3;
    double tolerance = -1.0e-18;
    PyGivenXConfig py_x;
  };

  explicit IBEdgeSelector(const Config& config);

  virtual ~IBEdgeSelector() = default;

  void setup(const ClusteringWorkspace& ws,
             const hydra::EmbeddingGroup& tasks,
             const hydra::EmbeddingDistance& metric) override;

  double scoreEdge(EdgeKey edge) override;

  bool updateFromEdge(EdgeKey edge) override;

  bool compareEdges(const std::pair<EdgeKey, double>& lhs,
                    const std::pair<EdgeKey, double>& rhs) const override;

  void onlineReweighting(double Ixy, double delta_weight) override;

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
  double I_xy_;
  double I_zy_prev_;
  double delta_weight_ = 1.0;
  std::vector<double> deltas_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EdgeSelector,
                                     IBEdgeSelector,
                                     IBEdgeSelector::Config>("IB");
};

void declare_config(IBEdgeSelector::Config& config);

}  // namespace clio
