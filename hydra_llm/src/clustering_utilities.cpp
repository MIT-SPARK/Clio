#include "hydra_lmm/clustering_utilities.h"

namespace hydra::llm {

double CosineDistance::norm(const Eigen::VectorXd& lhs,
                            const Eigen::VectorXd& rhs) const {
  const auto divisor = lhs.norm() * rhs.norm();
  return lhs.dot(rhs) / divisor;
}

double L2Norm::norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs - rhs).norm();
}

double L1Norm::norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const {
  return (lhs - rhs).lpNorm<1>();
}

// task_embeddings = [helpers.get_text_clip_feature(x) for x in tasks]
struct TaskEmbeddings {
  std::vector<Eigen::VectorXd> embeddings;
};

struct Clustering {
  struct Config {
    config::VirtualConfig<EmbeddingNorm> norm;
    double stop_value = 0.0;
  };

  Clustering(const Config& config)
      : config(config::checkValid(config)),
        norm_(config.norm.create()),
        norm(*CHECK_NOTNULL(norm_)) {}

  const Config config;
  std::unique_ptr<EmbeddingNorm> norm_;
  const EmbeddingNorm& norm;
  TaskEmbeddings tasks;

  Eigen::MatrixXd scoreMatrices(const std::vector<Eigen::VectorXd>& assignments) {
    Eigen::MatrixXd scores(assignments.size(), tasks.embeddings.size());
    for (size_t r = 0; r < assignments.size(); ++r) {
      for (size_t c = 0; c < tasks.embeddings.size(); ++c) {
        scores(r, c) = norm(tasks.embeddings[r], assignments[c]);
      }
    }

    return scores;
  }

  std::vector<double> computePhi(const SceneGraphLayer& layer,
                                 const Eigen::MatrixXd& scores) {
    // TODO(nathan) compute actual edge weights;
    // compute node phi via colwise max of dists
    // compute edge phi via averaging features and taking cosine distance with tasks
    // phi is then max(phi_e - phi_s, phi_e - phi_t)
    return std::vector<double>(layer.numEdges(), 0.0);
  }

  // clusters_all, sims = self.find_clusters(adj, features, tasks, stop)
  void cluster(const SceneGraphLayer& layer,
               const std::vector<Eigen::VectorXd>& nodes) {
    const auto dists = scoreMatrices(nodes);
    // compute edge features
    const auto phi = computePhi(layer, scores);
    std::vector<size_t> assignments(nodes.size());
    std::atoi(assignments.begin(), assignments.end(), 0);
    std::set<size_t> cluster_roots(assignments.begin(), assignments.end());

    for (size_t i = 0; i < assignments.size(); ++i) {
      // iter will always be valid: cluster roots is never empty
      auto iter = std::max_element(
          cluster_roots.begin(),
          cluster_roots.end(),
          [phi](const auto& lhs, const auto& rhs) { return phi[lhs] < phi[rhs]; });

      if (phi[*iter] <= config.stop_value) {
        break;
      }

      // record merge (including merging edge features)
      // compute new edges weights for merged node
    }

    // filter clusters below a certain threshold
    // return final clusters
  }
};

}  // namespace hydra::llm
