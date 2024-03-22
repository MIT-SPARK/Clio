#include "hydra_llm/ib_edge_selector.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/common.h>

#include "hydra_llm/common.h"
#include "hydra_llm/probability_utilities.h"

namespace hydra::llm {

using Indices = std::vector<std::pair<size_t, size_t>>;

size_t numFeatures(const NodeEmbeddingMap& features) { return features.size(); }

void declare_config(IBEdgeSelector::Config& config) {
  using namespace config;
  name("IBEdgeSelector::Config");
  field(config.max_delta, "max_delta");
  field(config.tolerance, "tolerance");
  field(config.py_x.score_threshold, "score_threshold");
  field(config.py_x.top_k, "top_k");
  field(config.py_x.cumulative, "cumulative");
  field(config.py_x.null_task_preprune, "null_task_preprune");

  check(config.max_delta, GE, 0.0, "max_delta");
  check(config.tolerance, LE, 0.0, "tolerance");
  check(config.py_x.top_k, GT, 0, "top_k");
}

IBEdgeSelector::IBEdgeSelector(const Config& config)
    : config(config::checkValid(config)) {}

void IBEdgeSelector::setup(const ClusteringWorkspace& ws,
                           const EmbeddingGroup& tasks,
                           const EmbeddingDistance& metric) {
  const auto fmt = getDefaultFormat();
  size_t N = ws.size();

  // p(z) = p(x) initially
  px_ = computeIBpx(ws);
  pz_ = px_;
  // p(z|x) is identity
  pz_x_ = Eigen::MatrixXd::Identity(N, N);

  py_x_ = computeIBpyGivenX(ws, tasks, metric, config.py_x);

  // p(y|z) = p(y|x) (as p(z) = p(x) and p(z|x) = I_n
  py_z_ = py_x_;
  // p(y) is uniform
  py_ = computeIBpy(tasks);

  VLOG(10) << "p(x): " << px_.format(fmt);
  VLOG(10) << "p(z): " << pz_.format(fmt);
  VLOG(10) << "p(y): " << py_.format(fmt);
  VLOG(10) << "p(y|x): " << py_x_.format(fmt);
  VLOG(10) << "p(y|z): " << py_z_.format(fmt);
  VLOG(10) << "p(z|x): " << pz_x_.format(fmt);

  // initialize mutual information to starting values;
  I_xy_ = mutualInformation(py_, px_, py_x_);
  I_zy_prev_ = I_xy_;
  deltas_.clear();
}

double IBEdgeSelector::scoreEdge(EdgeKey edge) {
  const auto p_s = pz_(edge.k1);
  const auto p_t = pz_(edge.k2);
  const auto total = p_s + p_t;
  Eigen::VectorXd prior(2);
  prior << p_s / total, p_t / total;
  Eigen::MatrixXd py_z_local(2, 2);
  py_z_local.col(0) = py_z_.col(edge.k1);
  py_z_local.col(1) = py_z_.col(edge.k2);
  return total * jensenShannonDivergence(py_z_local, prior);
}

bool IBEdgeSelector::updateFromEdge(EdgeKey edge) {
  // we merge target -> source
  const auto p_s = pz_(edge.k1);
  const auto p_t = pz_(edge.k2);
  // update new cluster probabilities
  pz_(edge.k1) = p_s + p_t;
  py_z_.col(edge.k1) =
      ((p_s * py_z_.col(edge.k1) + p_t * py_z_.col(edge.k2)) / (p_s + p_t)).eval();
  pz_x_.col(edge.k1) += pz_x_.col(edge.k2);

  // zero-out merged nodes
  pz_(edge.k2) = 0.0;
  py_z_.col(edge.k2).setConstant(0.0);
  pz_x_.col(edge.k2).setConstant(0.0);

  // for I[a; b] order is p(a), p(b), p(a|b)
  const auto I_zy = mutualInformation(py_, pz_, py_z_);
  const auto d_I_zy = I_zy_prev_ - I_zy;

  // avoid divide-by-zero and other weirdness with precision
  const auto delta = delta_weight_ * d_I_zy / I_xy_;

  I_zy_prev_ = I_zy;
  deltas_.push_back(delta);
  return delta >= config.max_delta;
}

bool IBEdgeSelector::compareEdges(const std::pair<EdgeKey, double>& lhs,
                                  const std::pair<EdgeKey, double>& rhs) const {
  return lhs.second < rhs.second;
}

void IBEdgeSelector::onlineReweighting(double Ixy, double delta_weight) {
  I_xy_ = Ixy;
  delta_weight_ = delta_weight;
}

std::string IBEdgeSelector::summarize() const {
  std::stringstream ss;
  ss << deltas_.size() - 1 << " merges, "
     << "δ_0=" << deltas_.front() << ", δ_n=" << deltas_.back();
  return ss.str();
}

}  // namespace hydra::llm
