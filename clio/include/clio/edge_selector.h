#pragma once
#include <hydra/openset/embedding_distances.h>
#include <hydra/openset/embedding_group.h>

#include "clio/clustering_workspace.h"
#include "clio/scene_graph_types.h"

namespace clio {

class EdgeSelector {
 public:
  using ScoreFunc = std::function<double(const Eigen::VectorXd&)>;
  using Ptr = std::unique_ptr<EdgeSelector>;

  virtual ~EdgeSelector() = default;

  virtual void setup(const ClusteringWorkspace& ws,
                     const hydra::EmbeddingGroup& tasks,
                     const hydra::EmbeddingDistance& metric) = 0;

  virtual double scoreEdge(EdgeKey edge) = 0;

  virtual bool updateFromEdge(EdgeKey edge) = 0;

  // compares two weighted edges (default should usually be w_lhs < w_rhs)
  virtual bool compareEdges(const std::pair<EdgeKey, double>& lhs,
                            const std::pair<EdgeKey, double>& rhs) const = 0;

  virtual void onlineReweighting(double param1, double param2) = 0;

  virtual std::string summarize() const = 0;
};

}  // namespace clio
