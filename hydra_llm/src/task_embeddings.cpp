#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

bool TaskEmbeddings::empty() const { return embeddings.empty(); }

Eigen::VectorXd TaskEmbeddings::getScores(const EmbeddingNorm& norm,
                                          const Eigen::VectorXd& embedding) const {
  Eigen::VectorXd distances(embeddings.size());
  for (size_t i = 0; i < embeddings.size(); ++i) {
    distances(i) = norm(embeddings[i], embedding);
  }

  return distances;
}

double TaskEmbeddings::getBestScore(const EmbeddingNorm& norm,
                                    const Eigen::VectorXd& embedding) const {
  double max_score = std::numeric_limits<double>::lowest();
  for (size_t i = 0; i < embeddings.size(); ++i) {
    const auto score = norm.score(embeddings[i], embedding);
    if (score > max_score) {
      max_score = score;
    }
  }

  return max_score;
}

}  // namespace hydra::llm
