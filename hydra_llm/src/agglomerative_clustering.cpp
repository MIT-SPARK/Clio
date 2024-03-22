#include "hydra_llm/agglomerative_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/utils/disjoint_set.h>
#include <hydra/utils/display_utilities.h>

#include <numeric>

#include "hydra_llm/edge_selector.h"

namespace hydra::llm {

void declare_config(AgglomerativeClustering::Config& config) {
  using namespace config;
  name("Clustering::Config");
  base<Clustering::Config>(config);
  field(config.metric, "metric");
  field(config.selector, "selector");
}

void clusterAgglomerative(ClusteringWorkspace& ws,
                          const EmbeddingGroup& tasks,
                          EdgeSelector& edge_selector,
                          const EmbeddingDistance& metric,
                          bool reweight,
                          double I_xy,
                          double delta_weight) {
  LOG(INFO) << "[IB] starting clustering with " << ws.edges.size() << " edges";

  edge_selector.setup(ws, tasks, metric);

  if (reweight) {
    edge_selector.onlineReweighting(I_xy, delta_weight);
  }

  for (auto& [edge, weight] : ws.edges) {
    weight = edge_selector.scoreEdge(edge);
  }

  for (size_t i = 0; i < ws.size(); ++i) {
    if (ws.edges.empty()) {
      // shouldn't happen unless |connected components| > 1
      break;
    }

    // iter will always be valid: always at least one edge
    auto best_edge_ptr = std::min_element(
        ws.edges.begin(), ws.edges.end(), [&](const auto& lhs, const auto& rhs) {
          return edge_selector.compareEdges(lhs, rhs);
        });
    CHECK(best_edge_ptr != ws.edges.end());

    const EdgeKey best_edge = best_edge_ptr->first;
    if (!edge_selector.updateFromEdge(best_edge)) {
      // we've hit a stop criteria
      break;
    }

    const auto changed_edges = ws.addMerge(best_edge);
    for (const auto edge : changed_edges) {
      ws.edges[edge] = edge_selector.scoreEdge(edge);
    }
  }

  LOG(INFO) << "[IB] " << edge_selector.summarize();
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
  clusterAgglomerative(ws, *tasks_, *edge_selector_, *metric_);

  const auto to_return = getClusters(ws, features);
  LOG(INFO) << "[IB] finished clustering with " << to_return.size() << " clusters";
  return to_return;
}

Clusters AgglomerativeClustering::getClusters(const ClusteringWorkspace& ws,
                                              const NodeEmbeddingMap& features) const {
  const auto cluster_nodes = ws.getClusters();

  Clusters to_return;
  for (const auto& nodes : cluster_nodes) {
    auto& cluster = to_return.emplace_back(std::make_shared<Cluster>());
    cluster->nodes.insert(nodes.begin(), nodes.end());

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
