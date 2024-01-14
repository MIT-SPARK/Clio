#include "hydra_llm/embedding_group.h"

#include "hydra_llm/embedding_distances.h"

namespace hydra::llm {

using ScoreResult = EmbeddingGroup::ScoreResult;

EmbeddingGroup::~EmbeddingGroup() {}

bool EmbeddingGroup::empty() const { return embeddings.empty(); }

Eigen::VectorXd EmbeddingGroup::getDistances(const EmbeddingDistance& dist,
                                             const Eigen::VectorXd& embedding) const {
  Eigen::VectorXd distances(embeddings.size());
  for (size_t i = 0; i < embeddings.size(); ++i) {
    distances(i) = dist.dist(embeddings[i], embedding);
  }

  return distances;
}

Eigen::VectorXd EmbeddingGroup::getScores(const EmbeddingDistance& dist,
                                          const Eigen::VectorXd& embedding) const {
  Eigen::VectorXd scores(embeddings.size());
  for (size_t i = 0; i < embeddings.size(); ++i) {
    scores(i) = dist.score(embeddings[i], embedding);
  }

  return scores;
}

ScoreResult EmbeddingGroup::getBestScore(const EmbeddingDistance& dist,
                                         const Eigen::VectorXd& embedding) const {
  ScoreResult result;
  for (size_t i = 0; i < embeddings.size(); ++i) {
    const auto score = dist.score(embeddings[i], embedding);
    if (score > result.score) {
      result.score = score;
      result.index = i;
    }
  }

  return result;
}

}  // namespace hydra::llm
