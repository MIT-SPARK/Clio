#include "hydra_llm/requery_edge_selector.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

namespace hydra::llm {

void declare_config(RequeryEdgeSelector::Config& config) {
  using namespace config;
  name("RequeryEdgeSelector::Config");
  field(config.stop_value, "stop_value");
  field(config.use_weighted_merge, "use_weighted_merge");
}

RequeryEdgeSelector::RequeryEdgeSelector(const Config& config)
    : config(config::checkValid(config)) {}

void RequeryEdgeSelector::setup(const ClusteringWorkspace& ws,
                                const ScoreFunc& score_func) {
  X = Eigen::MatrixXd(ws.featureDim(), ws.size());
  scores.resize(ws.size());
  for (auto&& [id, feature] : ws.features) {
    X.col(id) = *feature;
    scores[id] = score_func(*feature);
  }
}

Eigen::VectorXd RequeryEdgeSelector::getMergedFeature(EdgeKey edge) const {
  const auto w1 = config.use_weighted_merge ? scores.at(edge.k1) : 1.0;
  const auto w2 = config.use_weighted_merge ? scores.at(edge.k2) : 1.0;
  return (w1 * X.col(edge.k1) + w2 * X.col(edge.k2)) / (w1 + w2);
}

double RequeryEdgeSelector::scoreEdge(const ClusteringWorkspace& /* ws */,
                                      const ScoreFunc& score_func,
                                      EdgeKey edge) {
  const auto c_n_1 = scores.at(edge.k1);
  const auto c_n_2 = scores.at(edge.k2);
  const auto x_e = getMergedFeature(edge);
  const auto phi_e = score_func(x_e);
  return std::max(phi_e - c_n_1, phi_e - c_n_2);
}

bool RequeryEdgeSelector::updateFromEdge(const ClusteringWorkspace& /* ws */,
                                         const ScoreFunc& score_func,
                                         EdgeKey edge) {
  // we follow the convention target -> source for merging
  const auto x_e = getMergedFeature(edge);
  const auto phi_e = score_func(x_e);
  if (phi_e <= config.stop_value) {
    return false;
  }

  X.col(edge.k1) = x_e;
  scores[edge.k1] = phi_e;
  return true;
}

bool RequeryEdgeSelector::compareEdges(const std::pair<EdgeKey, double>& lhs,
                                       const std::pair<EdgeKey, double>& rhs) const {
  // this is inverted (requery takes max edge)
  return lhs.second > rhs.second;
}

std::string RequeryEdgeSelector::summarize() const { return ""; }

}  // namespace hydra::llm
