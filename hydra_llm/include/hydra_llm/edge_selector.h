#pragma once
#include "hydra_llm/clustering_workspace.h"
#include "hydra_llm/embedding_distances.h"
#include "hydra_llm/embedding_group.h"

namespace hydra::llm {

class EdgeSelector {
 public:
  using ScoreFunc = std::function<double(const Eigen::VectorXd&)>;
  using Ptr = std::unique_ptr<EdgeSelector>;

  virtual ~EdgeSelector() = default;

  virtual void setup(const ClusteringWorkspace& ws,
                     const EmbeddingGroup& tasks,
                     const EmbeddingDistance& metric) = 0;

  virtual double scoreEdge(EdgeKey edge) = 0;

  virtual bool updateFromEdge(EdgeKey edge) = 0;

  // compares two weighted edges (default should usually be w_lhs < w_rhs)
  virtual bool compareEdges(const std::pair<EdgeKey, double>& lhs,
                            const std::pair<EdgeKey, double>& rhs) const = 0;

  virtual void onlineReweighting(double param1, double param2) = 0;

  virtual std::string summarize() const = 0;
};

}  // namespace hydra::llm
