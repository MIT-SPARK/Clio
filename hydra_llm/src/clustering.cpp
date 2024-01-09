#include "hydra_llm/clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/utils/disjoint_set.h>
#include <hydra/utils/display_utilities.h>

#include <numeric>

namespace hydra::llm {

void declare_config(Clustering::Config& config) {
  using namespace config;
  name("Clustering::Config");
  field(config.tasks, "tasks");
  field(config.norm, "norm");
  field(config.merge, "merge");
  field(config.stop_value, "stop_value");
  field(config.min_score, "min_score");
}

struct ClipNodeAttributes : public NodeAttributes {
  double score;
  const ClipEmbedding* clip;

  virtual NodeAttributes::Ptr clone() const {
    return std::make_unique<ClipNodeAttributes>(*this);
  }
};

Clustering::Clustering(const Config& config)
    : config(config::checkValid(config)),
      norm_(config.norm.create()),
      embedding_merge_(config.merge.create()),
      tasks_(config.tasks.create()),
      norm(*CHECK_NOTNULL(norm_)) {}

void Clustering::computePhi(const SceneGraphLayer& layer,
                            const std::set<EdgeKey>& edges,
                            EdgeEmbeddingMap& phi) const {
  for (const auto edge : edges) {
    const auto& attrs1 = layer.getNode(edge.k1)->get().attributes<ClipNodeAttributes>();
    const auto& attrs2 = layer.getNode(edge.k2)->get().attributes<ClipNodeAttributes>();
    const auto phi_1 = attrs1.score;
    const auto phi_2 = attrs2.score;
    auto new_clip = embedding_merge_->merge(*attrs1.clip, phi_1, *attrs1.clip, phi_2);

    const auto phi_e = tasks_->getBestScore(norm, new_clip->embedding);
    const auto weight = std::max(phi_e - phi_1, phi_e - phi_2);
    phi.emplace(edge, EdgeEmbeddingInfo{weight, std::move(new_clip)});
  }
}

void Clustering::fillSubgraph(const SceneGraphLayer& layer,
                              const NodeEmbeddingMap& node_embeddings,
                              IsolatedSceneGraphLayer& cluster_layer) const {
  for (auto&& [node_id, clip] : node_embeddings) {
    if (!clip) {
      LOG(WARNING) << "Node '" << printNodeId(node_id) << "' missing clip feature";
      continue;
    }

    const auto& node = layer.getNode(node_id)->get();
    auto attrs = std::make_unique<ClipNodeAttributes>();
    attrs->position = node.attributes().position;
    attrs->score = tasks_->getBestScore(norm, clip->embedding);
    attrs->clip = clip;
    cluster_layer.emplaceNode(node_id, std::move(attrs));

    // this will get us all edges between nodes in the subgraph
    for (const auto& sibling : node.siblings()) {
      cluster_layer.insertEdge(node_id, sibling);
    }
  }
}

bool keysIntersect(EdgeKey key1, EdgeKey key2) {
  return key1.k1 == key2.k1 || key1.k1 == key2.k2 || key1.k2 == key2.k1 ||
         key1.k2 == key2.k2;
}

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

using Clusters = std::vector<Cluster::Ptr>;

Clusters Clustering::cluster(const SceneGraphLayer& layer,
                             const NodeEmbeddingMap& node_embeddings) const {
  IsolatedSceneGraphLayer cluster_layer(layer.id);
  fillSubgraph(layer, node_embeddings, cluster_layer);

  std::set<EdgeKey> all_edges;
  for (const auto& key_edge_pair : cluster_layer.edges()) {
    all_edges.insert(key_edge_pair.first);
  }

  // populate possible merges
  EdgeEmbeddingMap phi;
  computePhi(layer, all_edges, phi);

  DisjointSet clusters(cluster_layer);
  std::map<NodeId, ClipEmbedding::Ptr> merged_embeddings;
  const auto potential_merges = cluster_layer.numNodes();
  for (size_t i = 0; i < potential_merges; ++i) {
    if (cluster_layer.numEdges() == 0) {
      // shouldn't happen unless |connected components| > 1
      break;
    }

    // iter will always be valid: always at least one edge
    auto iter =
        std::max_element(phi.begin(), phi.end(), [&](const auto& lhs, const auto& rhs) {
          return lhs.second.weight < rhs.second.weight;
        });

    if (phi.at(iter->first).weight <= config.stop_value) {
      break;
    }

    const auto key = iter->first;
    merged_embeddings.erase(key.k1);
    auto new_iter =
        merged_embeddings.emplace(key.k1, std::move(iter->second.clip)).first;

    cluster_layer.mergeNodes(key.k2, key.k1);  // prefer the lower node id
    cluster_layer.getNode(key.k1)->get().attributes<ClipNodeAttributes>().clip =
        new_iter->second.get();

    clusters.doUnion(key.k2, key.k1);

    const auto to_replace = pruneEdges(key, phi);
    const auto remapped_edges = remapEdges(clusters, to_replace);
    computePhi(layer, remapped_edges, phi);
  }

  // TODO(nathan) refactor out into another function
  Clusters to_return;
  std::map<NodeId, size_t> cluster_lookup;
  for (const auto& root : clusters.roots) {
    const auto& attrs = cluster_layer.getNode(root)->get().attributes<ClipNodeAttributes>();
    const auto score = tasks_->getBestScore(norm, attrs.clip->embedding);
    if (score < config.min_score) {
      continue;
    }

    auto new_cluster = std::make_shared<Cluster>();
    new_cluster->clip = std::make_unique<ClipEmbedding>(attrs.clip->embedding);
    new_cluster->score = score;
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

}  // namespace hydra::llm
