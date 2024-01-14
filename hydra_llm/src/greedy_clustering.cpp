#include "hydra_llm/greedy_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/utils/disjoint_set.h>
#include <hydra/utils/display_utilities.h>

#include <numeric>

namespace hydra::llm {

using Clusters = std::vector<Cluster::Ptr>;

void declare_config(GreedyClustering::Config& config) {
  using namespace config;
  name("Clustering::Config");
  base<Clustering::Config>(config);
  field(config.metric, "metric");
  field(config.merge, "merge");
  field(config.stop_value, "stop_value");
  field(config.min_score, "min_score");
}

bool keysIntersect(EdgeKey key1, EdgeKey key2) {
  return key1.k1 == key2.k1 || key1.k1 == key2.k2 || key1.k2 == key2.k1 ||
         key1.k2 == key2.k2;
}

struct ClusteringWorkspace {
  DisjointSet clusters;
  std::map<NodeId, ScoredEmbedding> embeddings;
  std::map<EdgeKey, ScoredEmbedding> edge_embeddings;
  std::map<EdgeKey, double> phi;

  ClusteringWorkspace(const EmbeddingGroup& tasks,
                      const EmbeddingMerger& merger,
                      const EmbeddingDistance& metric,
                      const SceneGraphLayer& layer,
                      const NodeEmbeddingMap& node_embeddings) {
    for (auto&& [node_id, clip] : node_embeddings) {
      if (!clip) {
        LOG(WARNING) << "Node '" << NodeSymbol(node_id).getLabel()
                     << "' missing clip feature";
        continue;
      }

      clusters.addSet(node_id);
      const auto result = tasks.getBestScore(metric, clip->embedding);
      embeddings.emplace(
          node_id,
          ScoredEmbedding{result.score,
                          std::make_unique<ClipEmbedding>(clip->embedding),
                          result.index});
    }

    for (const auto node_id : clusters.roots) {
      const auto& node = layer.getNode(node_id)->get();
      for (const auto& sibling : node.siblings()) {
        if (!embeddings.count(sibling)) {
          continue;
        }

        const EdgeKey new_edge{node_id, sibling};
        if (edge_embeddings.count(new_edge)) {
          continue;  // undirected edges: we need to skip duplicates
        }

        addEdge(tasks, merger, metric, new_edge);
      }
    }
  }

  size_t size() const { return clusters.roots.size(); }

  size_t numMergeCandidates() const { return edge_embeddings.size(); }

  void addEdge(const EmbeddingGroup& tasks,
               const EmbeddingMerger& merger,
               const EmbeddingDistance& metric,
               EdgeKey edge) {
    auto n1_iter = embeddings.find(edge.k1);
    auto n2_iter = embeddings.find(edge.k2);
    CHECK(n1_iter != embeddings.end())
        << "Missing: '" << NodeSymbol(edge.k1).getLabel() << "'";
    CHECK(n2_iter != embeddings.end())
        << "Missing: '" << NodeSymbol(edge.k2).getLabel() << "'";
    const auto& n1 = n1_iter->second;
    const auto& n2 = n2_iter->second;
    const auto phi_1 = n1.score;
    const auto phi_2 = n2.score;
    auto new_clip = merger.merge(*n1.clip, phi_1, *n2.clip, phi_2);

    const auto result = tasks.getBestScore(metric, new_clip->embedding);
    const auto phi_e = result.score;
    edge_embeddings.emplace(edge,
                            ScoredEmbedding{phi_e, std::move(new_clip), result.index});
    phi.emplace(edge, std::max(phi_e - phi_1, phi_e - phi_2));
  }

  std::pair<EdgeKey, double> getMergeCandidate() const {
    // iter will always be valid: always at least one edge
    auto iter = std::max_element(edge_embeddings.begin(),
                                 edge_embeddings.end(),
                                 [&](const auto& lhs, const auto& rhs) {
                                   return phi.at(lhs.first) < phi.at(rhs.first);
                                 });
    CHECK(iter != edge_embeddings.end());
    return {iter->first, phi.at(iter->first)};
  }

  std::set<EdgeKey> addMerge(EdgeKey key) {
    const auto erased_opt = clusters.doUnion(key.k2, key.k1);
    CHECK(erased_opt);  // we always merge two different clusters
    const auto erased = *erased_opt;
    const auto kept = erased == key.k1 ? key.k2 : key.k1;
    {  // limit scope for invalid references
      // copy merge candidate over to cluster
      auto& edge_info = edge_embeddings.at(key);
      auto& cluster_info = embeddings.at(kept);
      cluster_info.score = edge_info.score;
      cluster_info.clip = std::move(edge_info.clip);
      // erase old edges and embeddings
      phi.erase(key);
      edge_embeddings.erase(key);
      embeddings.erase(erased);
    }

    std::set<EdgeKey> to_replace;
    auto iter = edge_embeddings.begin();
    while (iter != edge_embeddings.end()) {
      if (!keysIntersect(key, iter->first)) {
        ++iter;
        continue;
      }

      const auto prev_key = iter->first;
      iter = edge_embeddings.erase(iter);
      phi.erase(prev_key);
      if (prev_key.k1 != erased && prev_key.k2 != erased) {
        to_replace.insert(prev_key);
        continue;
      }

      const EdgeKey new_edge{clusters.findSet(prev_key.k1),
                             clusters.findSet(prev_key.k2)};
      if (new_edge.k1 == new_edge.k2) {
        continue;
      }

      to_replace.insert(new_edge);
    }

    return to_replace;
  }

  void updatePhi(const EmbeddingGroup& tasks,
                 const EmbeddingMerger& merger,
                 const EmbeddingDistance& metric,
                 const std::set<EdgeKey>& edges) {
    for (const auto& edge : edges) {
      addEdge(tasks, merger, metric, edge);
    }
  }

  Clusters getClusters(const EmbeddingGroup& tasks, double min_score) {
    Clusters to_return;
    std::map<NodeId, size_t> cluster_lookup;
    for (auto&& [root, info] : embeddings) {
      if (info.score < min_score) {
        continue;
      }

      auto new_cluster = std::make_shared<Cluster>();
      new_cluster->clip = std::make_unique<ClipEmbedding>(info.clip->embedding);
      new_cluster->score = info.score;
      new_cluster->best_task_index = info.task_index;
      new_cluster->best_task_name = tasks.tasks.at(info.task_index);
      cluster_lookup[root] = to_return.size();
      to_return.push_back(new_cluster);
    }

    for (auto&& [node, parent] : clusters.parents) {
      const auto root = clusters.findSet(parent);
      auto iter = cluster_lookup.find(root);
      if (iter == cluster_lookup.end()) {
        continue;
      }

      to_return[iter->second]->nodes.insert(node);
    }

    return to_return;
  }
};

GreedyClustering::GreedyClustering(const Config& config)
    : Clustering(config),
      config(config::checkValid(config)),
      metric_(config.metric.create()),
      embedding_merge_(config.merge.create()) {}

Clusters GreedyClustering::cluster(const SceneGraphLayer& layer,
                                   const NodeEmbeddingMap& features) const {
  if (tasks_->empty()) {
    LOG(ERROR) << "No tasks present: cannot cluster";
    return {};
  }

  ClusteringWorkspace workspace(*tasks_, *embedding_merge_, *metric_, layer, features);

  const auto potential_merges = workspace.size();
  for (size_t i = 0; i < potential_merges; ++i) {
    if (workspace.numMergeCandidates() == 0) {
      // shouldn't happen unless |connected components| > 1
      break;
    }

    auto&& [key, candidate_score] = workspace.getMergeCandidate();
    if (candidate_score <= config.stop_value) {
      break;
    }

    const auto changed_edges = workspace.addMerge(key);
    workspace.updatePhi(*tasks_, *embedding_merge_, *metric_, changed_edges);
  }

  return workspace.getClusters(*tasks_, config.min_score);
}

}  // namespace hydra::llm
