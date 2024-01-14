#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace hydra::llm {

struct EmbeddingDistance;

struct EmbeddingGroup {
  using Ptr = std::shared_ptr<EmbeddingGroup>;
  struct ScoreResult {
    double score = std::numeric_limits<double>::lowest();
    size_t index = 0;
  };

  virtual ~EmbeddingGroup();

  bool empty() const;
  operator bool() const { return !empty(); }

  Eigen::VectorXd getDistances(const EmbeddingDistance& dist,
                               const Eigen::VectorXd& embedding) const;

  Eigen::VectorXd getScores(const EmbeddingDistance& dist,
                            const Eigen::VectorXd& embedding) const;

  ScoreResult getBestScore(const EmbeddingDistance& dist,
                           const Eigen::VectorXd& embedding) const;

  std::vector<Eigen::VectorXd> embeddings;
  std::vector<std::string> tasks;
};

}  // namespace hydra::llm
