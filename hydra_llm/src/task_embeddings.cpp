#include "hydra_llm/task_embeddings.h"

namespace hydra::llm {

using ScoreResult = TaskEmbeddings::ScoreResult;

bool TaskEmbeddings::empty() const { return embeddings.empty(); }

Eigen::VectorXd TaskEmbeddings::getScores(const EmbeddingNorm& norm,
                                          const Eigen::VectorXd& embedding) const {
  Eigen::VectorXd distances(embeddings.size());
  for (size_t i = 0; i < embeddings.size(); ++i) {
    distances(i) = norm(embeddings[i], embedding);
  }

  return distances;
}

ScoreResult TaskEmbeddings::getBestScore(const EmbeddingNorm& norm,
                                         const Eigen::VectorXd& embedding) const {
  ScoreResult result;
  for (size_t i = 0; i < embeddings.size(); ++i) {
    const auto score = norm.score(embeddings[i], embedding);
    if (score > result.score) {
      result.score = score;
      result.index = i;
    }
  }

  return result;
}

}  // namespace hydra::llm
