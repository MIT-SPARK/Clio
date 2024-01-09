#pragma once
#include <Eigen/Dense>

#include "hydra_llm/embedding_norms.h"

namespace hydra::llm {

// task_embeddings = [helpers.get_text_clip_feature(x) for x in tasks]
struct TaskEmbeddings {
  using Ptr = std::shared_ptr<TaskEmbeddings>;

  virtual ~TaskEmbeddings() = default;

  bool empty() const;
  operator bool() const { return !empty(); }

  Eigen::VectorXd getScores(const EmbeddingNorm& norm,
                            const Eigen::VectorXd& embedding) const;

  double getBestScore(const EmbeddingNorm& norm,
                      const Eigen::VectorXd& embedding) const;

  std::vector<Eigen::VectorXd> embeddings;
  std::vector<std::string> tasks;
};

}  // namespace hydra::llm
