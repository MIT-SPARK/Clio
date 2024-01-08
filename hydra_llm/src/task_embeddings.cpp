#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

bool TaskEmbeddings::empty() const { return embeddings.cols() == 0; }

Eigen::VectorXd TaskEmbeddings::getDistances(const EmbeddingNorm& norm,
                                             const Eigen::VectorXd& embedding) const {
  Eigen::VectorXd distances(embeddings.cols());
  for (int i = 0; i < embeddings.cols(); ++i) {
    distances(i) = norm(embeddings.col(i), embedding);
  }

  return distances;
}

double TaskEmbeddings::getBestDistance(const EmbeddingNorm& norm,
                                       const Eigen::VectorXd& embedding) const {
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < embeddings.cols(); ++i) {
    const auto dist = norm(embeddings.col(i), embedding);
    if (dist < min_dist) {
      min_dist = dist;
    }
  }

  return min_dist;
}

}  // namespace hydra::llm
