#include "hydra_llm/agglomerative_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/utils/disjoint_set.h>
#include <hydra/utils/display_utilities.h>

#include <numeric>

#include "hydra_llm/clustering_workspace.h"
#include "hydra_llm/edge_selector.h"

namespace hydra::llm {

void declare_config(AgglomerativeClustering::Config& config) {
  using namespace config;
  name("Clustering::Config");
  base<Clustering::Config>(config);
  field(config.metric, "metric");
  field(config.selector, "selector");
}

AgglomerativeClustering::AgglomerativeClustering(const Config& config)
    : Clustering(config),
      config(config::checkValid(config)),
      metric_(config.metric.create()),
      edge_selector_(config.selector.create()) {}

Clusters AgglomerativeClustering::cluster(const SceneGraphLayer& layer,
                                          const NodeEmbeddingMap& features) const {
  if (tasks_->empty()) {
    LOG(ERROR) << "No tasks present: cannot cluster";
    return {};
  }

  ClusteringWorkspace ws(layer, features);
  const auto f_score = [this](const Eigen::VectorXd& x) {
    return tasks_->getBestScore(*metric_, x).score;
  };
  LOG(INFO) << "[IB] starting clustering with " << ws.edges.size() << " edges";

  edge_selector_->setup(ws, f_score);

  for (auto& [edge, weight] : ws.edges) {
    weight = edge_selector_->scoreEdge(ws, f_score, edge);
  }

  for (size_t i = 0; i < ws.size(); ++i) {
    if (ws.edges.empty()) {
      // shouldn't happen unless |connected components| > 1
      break;
    }

    // iter will always be valid: always at least one edge
    auto best_edge_ptr = std::min_element(
        ws.edges.begin(), ws.edges.end(), [&](const auto& lhs, const auto& rhs) {
          return edge_selector_->compareEdges(lhs, rhs);
        });
    CHECK(best_edge_ptr != ws.edges.end());

    const EdgeKey best_edge = best_edge_ptr->first;
    if (!edge_selector_->updateFromEdge(ws, f_score, best_edge)) {
      // we've hit a stop criteria
      break;
    }

    const auto changed_edges = ws.addMerge(best_edge);
    for (const auto edge : changed_edges) {
      ws.edges[edge] = edge_selector_->scoreEdge(ws, f_score, edge);
    }
  }

  const auto to_return = getClusters(ws.assignments, ws.node_lookup, features);
  LOG(INFO) << "[IB] " << edge_selector_->summarize();
  LOG(INFO) << "[IB] finished clustering with " << to_return.size() << " clusters";
  return to_return;
}

Clusters AgglomerativeClustering::getClusters(
    const std::vector<size_t>& assignments,
    const std::map<size_t, NodeId>& node_lookup,
    const NodeEmbeddingMap& features) const {
  const std::set<size_t> cluster_ids(assignments.begin(), assignments.end());

  size_t index = 0;
  std::map<size_t, size_t> cluster_lookup;
  for (const auto cluster_id : cluster_ids) {
    cluster_lookup[cluster_id] = index;
    ++index;
  }

  Clusters to_return(cluster_ids.size());
  for (size_t i = 0; i < assignments.size(); ++i) {
    auto& cluster = to_return.at(cluster_lookup[assignments[i]]);
    if (!cluster) {
      cluster.reset(new Cluster());
    }

    cluster->nodes.insert(node_lookup.at(i));
  }

  for (const auto& cluster : to_return) {
    auto iter = cluster->nodes.begin();
    cluster->feature = features.at(*iter);
    ++iter;
    while (iter != cluster->nodes.end()) {
      cluster->feature += features.at(*iter);
      ++iter;
    }

    cluster->feature /= cluster->nodes.size();

    const auto info = tasks_->getBestScore(*metric_, cluster->feature);
    cluster->score = info.score;
    cluster->best_task_index = info.index;
    cluster->best_task_name = tasks_->tasks.at(info.index);
  }

  // TODO(nathan) filter by min score
  return to_return;
}

}  // namespace hydra::llm
