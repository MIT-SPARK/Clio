#include "hydra_llm/clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/utils/disjoint_set.h>
#include <hydra/utils/display_utilities.h>

#include <numeric>

namespace hydra::llm {

using Clusters = std::vector<Cluster::Ptr>;

void declare_config(Clustering::Config& config) {
  using namespace config;
  name("Clustering::Config");
  field(config.tasks, "tasks");
  field(config.norm, "norm");
  field(config.merge, "merge");
  field(config.stop_value, "stop_value");
  field(config.min_score, "min_score");
}

bool keysIntersect(EdgeKey key1, EdgeKey key2) {
  return key1.k1 == key2.k1 || key1.k1 == key2.k2 || key1.k2 == key2.k1 ||
         key1.k2 == key2.k2;
}

/*
std::vector<EdgeKey> pruneEdges(EdgeKey merge, EdgeEmbeddingMap& phi) {
  std::vector<EdgeKey> to_replace;
  auto iter = phi.begin();
  while (iter != phi.end()) {
    if (!keysIntersect(merge, iter->first)) {
      ++iter;
      continue;
    }

    to_replace.push_back(iter->first);
    iter = phi.erase(iter);
  }

  return to_replace;
}
*/

std::set<EdgeKey> remapEdges(const DisjointSet& clusters,
                             const std::vector<EdgeKey>& edges) {
  std::set<EdgeKey> to_return;
  for (const auto orig : edges) {
    const EdgeKey new_edge{clusters.findSet(orig.k1), clusters.findSet(orig.k2)};
    if (new_edge.k1 == new_edge.k2) {
      continue;
    }

    to_return.insert(new_edge);
  }

  return to_return;
}

struct ScoredEmbedding {
  double weight;
  ClipEmbedding::Ptr clip;
};

struct ClusteringWorkspace {
  DisjointSet clusters;
  std::map<NodeId, ScoredEmbedding> embeddings;
  std::map<EdgeKey, ScoredEmbedding> edge_embeddings;
  std::map<EdgeKey, double> phi;

  ClusteringWorkspace(const TaskEmbeddings& tasks,
                      const EmbeddingMerger& merger,
                      const EmbeddingNorm& norm,
                      const SceneGraphLayer& layer,
                      const NodeEmbeddingMap& node_embeddings) {
    for (auto&& [node_id, clip] : node_embeddings) {
      if (!clip) {
        LOG(WARNING) << "Node '" << printNodeId(node_id) << "' missing clip feature";
        continue;
      }

      clusters.addSet(node_id);
      const auto score = tasks.getBestScore(norm, clip->embedding);
      embeddings.emplace(
          node_id,
          ScoredEmbedding{score, std::make_unique<ClipEmbedding>(clip->embedding)});
    }

    for (const auto node_id : clusters.roots) {
      const auto& node = layer.getNode(node_id)->get();
      for (const auto& sibling : node.siblings()) {
        if (embeddings.count(sibling)) {
          continue;
        }

        const EdgeKey new_edge{node_id, sibling};
        if (edge_embeddings.count(new_edge)) {
          continue;  // handle undirected case
        }

        addEdge(tasks, merger, norm, new_edge);
      }
    }
  }

  size_t size() const { return clusters.roots.size(); }

  size_t numMergeCandidates() const { return edge_embeddings.size(); }

  void addEdge(const TaskEmbeddings& tasks,
               const EmbeddingMerger& merger,
               const EmbeddingNorm& norm,
               EdgeKey edge) {
    auto n1_iter = embeddings.find(edge.k1);
    auto n2_iter = embeddings.find(edge.k2);
    CHECK(n1_iter != embeddings.end()) << "Missing: '" << printNodeId(edge.k1) << "'";
    CHECK(n2_iter != embeddings.end()) << "Missing: '" << printNodeId(edge.k2) << "'";
    const auto& n1 = n1_iter->second;
    const auto& n2 = n2_iter->second;
    const auto phi_1 = n1.weight;
    const auto phi_2 = n2.weight;
    auto new_clip = merger.merge(*n1.clip, phi_1, *n2.clip, phi_2);

    const auto phi_e = tasks.getBestScore(norm, new_clip->embedding);
    edge_embeddings.emplace(edge, ScoredEmbedding{phi_e, std::move(new_clip)});
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

  void updatePhi(const TaskEmbeddings& tasks,
                 const EmbeddingMerger& merger,
                 const EmbeddingNorm& norm,
                 const std::set<EdgeKey>& edges) {
    for (const auto& edge : edges) {
      addEdge(tasks, merger, norm, edge);
    }
  }

  void addMerge(EdgeKey key) {
    const auto erased = clusters.doUnion(key.k2, key.k1);
    CHECK(erased); // we always merge two different clusters
    const auto kept = erased == key.k1 ? key.k2 : key.k1;
    { // limit scope for invalid references
      // copy merge candidate over to cluster
      auto& edge_info = edge_embeddings.at(key);
      auto& cluster_info = embeddings.at(kept);
      cluster_info.weight = edge_info.weight;
      cluster_info.clip = std::move(edge_info.clip);
      // erase old edges and embeddings
      phi.erase(key);
      edge_embeddings.erase(key);
      embeddings.erase(*erased);
    }

    // TODO(nathan) fix
    //const auto to_replace = pruneEdges(key, phi);
    //const auto remapped_edges = remapEdges(clusters, to_replace);
    //computePhi(layer, remapped_edges, phi);
  }

  Clusters getClusters(double min_score) {
    Clusters to_return;
    std::map<NodeId, size_t> cluster_lookup;
    for (auto&& [root, info] : embeddings) {
      if (info.weight < min_score) {
        continue;
      }

      auto new_cluster = std::make_shared<Cluster>();
      new_cluster->clip = std::make_unique<ClipEmbedding>(info.clip->embedding);
      new_cluster->score = info.weight;
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

Clustering::Clustering(const Config& config)
    : config(config::checkValid(config)),
      norm_(config.norm.create()),
      embedding_merge_(config.merge.create()),
      tasks_(config.tasks.create()),
      norm(*CHECK_NOTNULL(norm_)) {}

Clusters Clustering::cluster(const SceneGraphLayer& original_layer,
                             const NodeEmbeddingMap& node_embeddings) const {
  if (tasks_->empty()) {
    LOG(ERROR) << "No tasks present: cannot cluster";
    return {};
  }

  ClusteringWorkspace workspace(
      *tasks_, *embedding_merge_, norm, original_layer, node_embeddings);

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

    workspace.addMerge(key);
  }

  return workspace.getClusters(config.min_score);
}

}  // namespace hydra::llm
