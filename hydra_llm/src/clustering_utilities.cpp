#include "hydra_llm/clustering_utilities.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <numeric>

namespace hydra::llm {

std::vector<double> Clustering::computePhi(const SceneGraphLayer& layer,
                                           const Eigen::MatrixXd& scores) {
  // TODO(nathan) compute actual edge weights;
  // compute node phi via colwise max of dists
  // compute edge phi via averaging features and taking cosine distance with tasks
  // phi is then max(phi_e - phi_s, phi_e - phi_t)
  return std::vector<double>(layer.numEdges(), 0.0);
}

// clusters_all, sims = self.find_clusters(adj, features, tasks, stop)
void Clustering::cluster(const SceneGraphLayer& layer,
                         const std::vector<Eigen::VectorXd>& nodes) {
  const auto dists = scoreMatrices(nodes);
  // compute edge features
  const auto phi = computePhi(layer, dists);
  std::vector<size_t> assignments(nodes.size());
  std::iota(assignments.begin(), assignments.end(), 0);
  std::set<size_t> cluster_roots(assignments.begin(), assignments.end());

  for (size_t i = 0; i < assignments.size(); ++i) {
    // iter will always be valid: cluster roots is never empty
    auto iter = std::max_element(
        cluster_roots.begin(),
        cluster_roots.end(),
        [&](const auto& lhs, const auto& rhs) { return phi[lhs] < phi[rhs]; });

    if (phi[*iter] <= config.stop_value) {
      break;
    }

    // record merge (including merging edge features)
    // compute new edges weights for merged node
  }

  // filter clusters below a certain threshold
  // return final clusters
}

}  // namespace hydra::llm
