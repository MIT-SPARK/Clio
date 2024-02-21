#include "hydra_llm/ib_edge_selector.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/common.h>

#include "hydra_llm/common.h"
#include "hydra_llm/probability_utilities.h"

namespace hydra::llm {

size_t numFeatures(const NodeEmbeddingMap& features) { return features.size(); }

void declare_config(IBEdgeSelector::Config& config) {
  using namespace config;
  name("IBEdgeSelector::Config");
  field(config.score_threshold, "score_threshold");
  field(config.min_beta, "min_beta");
  field(config.tolerance, "tolerance");
  check(config.min_beta, GE, 0.0, "min_beta");
  check(config.tolerance, LE, 0.0, "tolerance");
}

IBEdgeSelector::IBEdgeSelector(const Config& config)
    : config(config::checkValid(config)) {}

void IBEdgeSelector::setup(const ClusteringWorkspace& ws, const ScoreFunc& score_func) {
  const auto fmt = getDefaultFormat();
  size_t N = ws.size();
  // p(x) is uniform, p(z) = p(x) initially
  px_ = Eigen::VectorXd::Constant(N, 1.0 / static_cast<double>(N));
  pz_ = px_;
  // p(z|x) is identity
  pz_x_ = Eigen::MatrixXd::Identity(N, N);

  // p(y=0|x) is approximated by min similarity
  py_x_ = Eigen::MatrixXd(2, N);
  py_x_.row(0).setConstant(config.score_threshold);
  // p(y=1|x) is approximated by score (clipped to be positive)
  for (auto&& [idx, feature] : ws.features) {
    py_x_(1, idx) = std::max(0.0, score_func(*feature));
  }
  VLOG(VLEVEL_DEBUG) << "raw: p(y|x): " << py_x_.format(fmt);
  double min = py_x_.row(1).minCoeff();
  double max = py_x_.row(1).maxCoeff();
  double avg = py_x_.row(1).sum() / ws.features.size();
  VLOG(VLEVEL_DEBUG) << "score average: " << avg << " (range: [" << min << ", " << max
                     << "]";

  const auto norm_factor = py_x_.colwise().sum();
  py_x_.array().rowwise() /= norm_factor.array();

  // p(y|z) = p(y|x) (as p(z) = p(x) and p(z|x) = I_n
  py_z_ = py_x_;
  // p(y) = p(y|x) * p(x)
  py_ = py_x_ * px_;

  VLOG(VLEVEL_DEBUG) << "p(x): " << px_.format(fmt);
  VLOG(VLEVEL_DEBUG) << "p(z): " << pz_.format(fmt);
  VLOG(VLEVEL_DEBUG) << "p(y): " << py_.format(fmt);
  VLOG(VLEVEL_DEBUG) << "p(y|x): " << py_x_.format(fmt);
  VLOG(VLEVEL_DEBUG) << "p(y|z): " << py_z_.format(fmt);
  VLOG(VLEVEL_DEBUG) << "p(z|x): " << pz_x_.format(fmt);

  // initialize mutual information to starting values;
  I_xz_prev_ = mutualInformation(pz_, px_, pz_x_);
  I_zy_prev_ = mutualInformation(py_, pz_, py_z_);
  VLOG(VLEVEL_DEBUG) << "start: I[z;x]=" << I_xz_prev_ << ", I[y;z]=" << I_zy_prev_;
  betas_.clear();
}

double IBEdgeSelector::scoreEdge(const ClusteringWorkspace&,
                                 const ScoreFunc&,
                                 EdgeKey edge) {
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

bool IBEdgeSelector::updateFromEdge(const ClusteringWorkspace& /*ws*/,
                                    const ScoreFunc& /*score_func*/,
                                    EdgeKey edge) {
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
  const auto I_xz = mutualInformation(pz_, px_, pz_x_);
  const auto I_zy = mutualInformation(py_, pz_, py_z_);

  // avoid divide-by-zero and other weirdness with precision
  const auto d_xz = I_xz - I_xz_prev_;
  const auto d_yz = std::min(I_zy - I_zy_prev_, config.tolerance);
  const auto beta = d_xz / d_yz;

  I_xz_prev_ = I_xz;
  I_zy_prev_ = I_zy;
  betas_.push_back(beta);
  return beta >= config.min_beta;
}

bool IBEdgeSelector::compareEdges(const std::pair<EdgeKey, double>& lhs,
                                  const std::pair<EdgeKey, double>& rhs) const {
  return lhs.second < rhs.second;
}

std::string IBEdgeSelector::summarize() const {
  std::stringstream ss;
  ss << betas_.size() - 1 << " merges, "
     << "β_0=" << betas_.front() << ", β_n=" << betas_.back();
  return ss.str();
}

}  // namespace hydra::llm
