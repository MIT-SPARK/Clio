#include "hydra_llm/places_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/common.h>
#include <hydra/common/hydra_config.h>
#include <hydra/rooms/room_utilities.h>
#include <hydra/utils/display_utilities.h>

namespace hydra::llm {

using Clusters = std::vector<Cluster::Ptr>;

void declare_config(PlaceClustering::Config& config) {
  using namespace config;
  name("PlaceClustering::Config");
  field(config.clustering, "clustering");
  field(config.min_assocation_iou, "min_assocation_iou");
  field(config.color_by_task, "color_by_task");
}

PlaceClustering::PlaceClustering(const Config& config)
    : config(config::checkValid(config)),
      clustering_(config.clustering.create()),
      region_id_('l', 0) {}

PlaceClustering::~PlaceClustering() {}

double computeIoU(const std::set<NodeId>& lhs, const std::set<NodeId>& rhs) {
  if (lhs.empty() && rhs.empty()) {
    return 0.0;
  }

  size_t num_same = 0;
  for (const auto& v : lhs) {
    if (rhs.count(v)) {
      ++num_same;
    }
  }

  const auto divisor = lhs.size() + rhs.size() - num_same;
  return static_cast<double>(num_same) / divisor;
}

void PlaceClustering::updateGraphIncremental(DynamicSceneGraph& graph,
                                             const NodeEmbeddingMap& views,
                                             const Clusters& clusters) const {
  const auto& places = graph.getLayer(DsgLayers::PLACES);
  std::map<NodeId, std::set<NodeId>> region_sets;
  for (const auto id_view_pair : views) {
    const auto node_id = id_view_pair.first;
    const SceneGraphNode& node = places.getNode(node_id)->get();
    const auto parent = node.getParent();
    if (parent) {
      auto iter = region_sets.find(*parent);
      if (iter == region_sets.end()) {
        iter = region_sets.emplace(*parent, std::set<NodeId>()).first;
      }

      iter->second.insert(node_id);
    }
  }

  std::map<size_t, NodeId> associations;
  for (size_t i = 0; i < clusters.size(); ++i) {
    const auto& cluster = clusters[i];
    double best_iou = std::numeric_limits<double>::lowest();
    for (auto&& [region, children] : region_sets) {
      const auto iou = computeIoU(cluster->nodes, children);
      if (iou < config.min_assocation_iou) {
        continue;
      }

      if (iou > best_iou) {
        associations[i] = region;
        best_iou = iou;
      }
    }
  }

  VLOG(VLEVEL_TRACE) << "Got " << clusters.size() << " cluster(s) with "
                     << associations.size() << " associations to " << region_sets.size()
                     << " previous clusters";

  // TODO(nathan) make region layer make semantic sense
  std::set<NodeId> updated_regions;
  std::vector<NodeId> new_regions;
  for (size_t i = 0; i < clusters.size(); ++i) {
    NodeId new_node_id;
    auto iter = associations.find(i);
    if (iter == associations.end()) {
      auto attrs = std::make_unique<RegionNodeAttributes>();
      attrs->semantic_label = 0;
      attrs->name = clusters[i]->best_task_name;
      attrs->semantic_feature = clusters[i]->clip->embedding;
      const auto color_idx =
          config.color_by_task ? clusters[i]->best_task_index : region_id_.categoryId();
      const auto color = HydraConfig::instance().getRoomColor(color_idx);
      attrs->color =
          Eigen::Map<const SemanticNodeAttributes::ColorVector>(color.data());
      graph.emplaceNode(DsgLayers::ROOMS, region_id_, std::move(attrs));
      new_node_id = region_id_;
      new_regions.push_back(region_id_);
      ++region_id_;
    } else {
      auto& attrs =
          graph.getNode(iter->second)->get().attributes<RegionNodeAttributes>();
      attrs.semantic_feature = clusters[i]->clip->embedding;
      new_node_id = iter->second;
    }

    for (const auto& node_id : clusters[i]->nodes) {
      const auto& node = graph.getNode(node_id)->get();
      const auto parent = node.getParent();
      if (parent) {
        updated_regions.insert(*parent);
        graph.removeEdge(*parent, node_id);
      }

      graph.insertEdge(new_node_id, node_id);
      updated_regions.insert(new_node_id);
    }
  }

  // TODO(nathan) this is ugly
  std::set<NodeId> to_delete;
  for (const auto node_id : updated_regions) {
    const auto& node = graph.getNode(node_id)->get();
    std::unordered_set<NodeId> to_use(node.children().begin(), node.children().end());
    if (to_use.empty()) {
      to_delete.insert(node_id);
      continue;
    }

    node.attributes().position = getRoomPosition(places, to_use);
    const auto siblings = node.siblings();
    for (const auto& sibling : siblings) {
      graph.removeEdge(node_id, sibling);
    }
  }

  VLOG(5) << "New: " << displayNodeSymbolContainer(new_regions);
  VLOG(5) << "Updated: " << displayNodeSymbolContainer(updated_regions);
  VLOG(5) << "Invalid: " << displayNodeSymbolContainer(to_delete);

  for (const auto node_id : to_delete) {
    graph.removeNode(node_id);
    updated_regions.erase(node_id);
  }

  addEdgesToRoomLayer(graph, updated_regions);

  const auto& regions = graph.getLayer(DsgLayers::ROOMS);
  for (auto&& [node_id, node] : regions.nodes()) {
    if (node->children().empty()) {
      LOG(ERROR) << "Invalid region: " << NodeSymbol(node_id).getLabel();
    }
  }
}

void PlaceClustering::updateGraphBatch(DynamicSceneGraph& graph,
                                       const Clusters& clusters) const {
  VLOG(VLEVEL_TRACE) << "Got " << clusters.size() << " cluster(s)";

  std::vector<NodeId> prev_regions;
  for (const auto& id_node_pair : graph.getLayer(DsgLayers::ROOMS).nodes()) {
    prev_regions.push_back(id_node_pair.first);
  }

  for (const auto node : prev_regions) {
    graph.removeNode(node);
  }

  std::set<NodeId> new_nodes;
  for (size_t i = 0; i < clusters.size(); ++i) {
    NodeSymbol new_node_id(region_id_.category(), i);
    auto attrs = std::make_unique<RegionNodeAttributes>();
    attrs->semantic_label = 0;
    attrs->name = clusters[i]->best_task_name;
    attrs->semantic_feature = clusters[i]->clip->embedding;
    const auto color_idx =
        config.color_by_task ? clusters[i]->best_task_index : region_id_.categoryId();
    const auto color = HydraConfig::instance().getRoomColor(color_idx);
    attrs->color = Eigen::Map<const SemanticNodeAttributes::ColorVector>(color.data());
    graph.emplaceNode(DsgLayers::ROOMS, new_node_id, std::move(attrs));

    for (const auto node_id : clusters[i]->nodes) {
      graph.insertEdge(new_node_id, node_id);
    }

    new_nodes.insert(new_node_id);
  }

  const auto& places = graph.getLayer(DsgLayers::PLACES);
  for (auto&& [node_id, node] : graph.getLayer(DsgLayers::ROOMS).nodes()) {
    const std::unordered_set<NodeId> to_use(node->children().begin(),
                                            node->children().end());
    node->attributes().position = getRoomPosition(places, to_use);
  }

  addEdgesToRoomLayer(graph, new_nodes);
}

void PlaceClustering::clusterPlaces(DynamicSceneGraph& graph,
                                    const NodeEmbeddingMap& views) {
  if (views.empty()) {
    LOG(ERROR) << "Need to have at least one view";
    return;
  }

  // TODO(nathan) maintain this at the updater func level?
  NodeEmbeddingMap valid_views;
  for (auto&& [node_id, view] : views) {
    const auto node = graph.getNode(node_id);
    if (!node) {
      continue;
    }

    auto& attrs = node->get().attributes<PlaceNodeAttributes>();
    if (view) {
      attrs.semantic_feature = view->embedding;
    }

    valid_views[node_id] = view;
  }

  const auto clusters =
      clustering_->cluster(graph.getLayer(DsgLayers::PLACES), valid_views);
  if (config.is_batch) {
    updateGraphBatch(graph, clusters);
  } else {
    updateGraphIncremental(graph, views, clusters);
  }
}

}  // namespace hydra::llm
