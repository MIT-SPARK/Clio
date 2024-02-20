#pragma once
#include <config_utilities/factory.h>

#include <Eigen/Dense>

#include "hydra_llm/edge_selector.h"

namespace hydra::llm {

class RequeryEdgeSelector : public EdgeSelector {
 public:
  struct Config {
    double stop_value = 0.0;
    bool use_weighted_merge = false;
  };

  explicit RequeryEdgeSelector(const Config& config);

  virtual ~RequeryEdgeSelector() = default;

  void setup(const ClusteringWorkspace& ws, const ScoreFunc& score_func) override;

  double scoreEdge(const ClusteringWorkspace& ws,
                   const ScoreFunc& score_func,
                   EdgeKey edge) override;

  bool updateFromEdge(const ClusteringWorkspace& ws,
                      const ScoreFunc& score_func,
                      EdgeKey edge) override;

  bool compareEdges(const std::pair<EdgeKey, double>& lhs,
                    const std::pair<EdgeKey, double>& rhs) const override;

  std::string summarize() const override;

  const Config config;

 private:
  Eigen::VectorXd getMergedFeature(EdgeKey edge) const;

  Eigen::MatrixXd X;
  std::vector<double> scores;

  inline static const auto registration_ =
      config::RegistrationWithConfig<EdgeSelector,
                                     RequeryEdgeSelector,
                                     RequeryEdgeSelector::Config>("Requery");
};

void declare_config(RequeryEdgeSelector::Config& config);

}  // namespace hydra::llm
