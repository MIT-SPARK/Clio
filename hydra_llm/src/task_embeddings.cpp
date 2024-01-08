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

}  // namespace hydra::llm
