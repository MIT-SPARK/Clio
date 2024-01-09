#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

bool TaskEmbeddings::empty() const { return embeddings.cols() == 0; }

Eigen::VectorXd TaskEmbeddings::getScores(const EmbeddingNorm& norm,
                                          const Eigen::VectorXd& embedding) const {
  Eigen::VectorXd distances(embeddings.cols());
  for (int i = 0; i < embeddings.cols(); ++i) {
    distances(i) = norm(embeddings.col(i), embedding);
  }

  return distances;
}

double TaskEmbeddings::getBestScore(const EmbeddingNorm& norm,
                                    const Eigen::VectorXd& embedding) const {
  double max_score = std::numeric_limits<double>::lowest();
  for (int i = 0; i < embeddings.cols(); ++i) {
    const auto score = norm.score(embeddings.col(i), embedding);
    if (score > max_score) {
      max_score = score;
    }
  }

  return max_score;
}

}  // namespace hydra::llm
