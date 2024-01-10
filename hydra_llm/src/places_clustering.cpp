#include "hydra_llm/places_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/common.h>
#include <hydra/common/hydra_config.h>
#include <hydra/rooms/room_utilities.h>
#include <hydra/utils/display_utilities.h>

namespace hydra::llm {

void declare_config(PlaceClustering::Config& config) {
  using namespace config;
  name("PlaceClustering::Config");
  field(config.clustering, "clustering");
  field(config.min_assocation_iou, "min_assocation_iou");
  field(config.color_by_task, "color_by_task");
}

const ClipView* getBestView(const std::map<size_t, ClipView::Ptr>& views,
                            const PlaceNodeAttributes& attrs) {
  double min_dist = std::numeric_limits<double>::max();
  const ClipView* best_view = nullptr;
  for (const auto& id_view_pair : views) {
    const auto& view = id_view_pair.second;
    const auto& sensor = *(view->sensor);

    const Eigen::Vector3f p_s = (view->sensor_T_world * attrs.position).cast<float>();
    if (!sensor.pointIsInViewFrustum(p_s)) {
      continue;
    }

    // heuristic to pick the view that's closest to the boundary of the free-space
    // sphere
    const auto dist = std::abs(attrs.distance - p_s.norm());
    if (dist < min_dist) {
      best_view = view.get();
      min_dist = dist;
    }
  }

  return best_view;
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

void PlaceClustering::clusterPlaces(DynamicSceneGraph& graph,
                                    const std::map<size_t, ClipView::Ptr>& views,
                                    const std::unordered_set<NodeId>& nodes) {
  if (views.empty()) {
    LOG(ERROR) << "Need to have at least one view";
    return;
  }

  std::map<NodeId, const ClipEmbedding*> assigned_views;
  const auto& places = graph.getLayer(DsgLayers::PLACES);
  std::map<NodeId, std::set<NodeId>> region_sets;
  for (const auto node_id : nodes) {
    const SceneGraphNode& node = places.getNode(node_id)->get();
    const auto& attrs = node.attributes<PlaceNodeAttributes>();
    const auto best_view = getBestView(views, attrs);
    if (!best_view) {
      continue;
    }

    assigned_views[node_id] = best_view->clip.get();
    const auto parent = node.getParent();
    if (parent) {
      auto iter = region_sets.find(*parent);
      if (iter == region_sets.end()) {
        iter = region_sets.emplace(*parent, std::set<NodeId>()).first;
      }

      iter->second.insert(node_id);
    }
  }

  VLOG(1) << "Using " << assigned_views.size() << " places of " << nodes.size()
          << " original places";
  const auto clusters = clustering_->cluster(places, assigned_views);

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

  LOG(WARNING) << "Got " << clusters.size() << " cluster(s) with "
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
      attrs->embedding = clusters[i]->clip->embedding;
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
      attrs.embedding = clusters[i]->clip->embedding;
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

  VLOG(VLEVEL_DEBUG) << "New: " << displayNodeSymbolContainer(new_regions);
  VLOG(VLEVEL_DEBUG) << "Updated: " << displayNodeSymbolContainer(updated_regions);
  VLOG(VLEVEL_DEBUG) << "Invalid: " << displayNodeSymbolContainer(to_delete);

  for (const auto node_id : to_delete) {
    graph.removeNode(node_id);
    updated_regions.erase(node_id);
  }

  addEdgesToRoomLayer(graph, updated_regions);

  const auto& regions = graph.getLayer(DsgLayers::ROOMS);
  for (auto&& [node_id, node] : regions.nodes()) {
    if (node->children().empty()) {
      LOG(ERROR) << "Invalid region: " << printNodeId(node_id);
    }
  }
}

}  // namespace hydra::llm
