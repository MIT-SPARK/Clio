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

  Eigen::VectorXd getDistances(const EmbeddingNorm& norm,
                               const Eigen::VectorXd& embedding) const;

  //! embeddings are row vectors -> embeddings.col(i) is embedding for task i
  Eigen::MatrixXd embeddings;
  //! task names
  std::vector<std::string> tasks;
};

}  // namespace hydra::llm
