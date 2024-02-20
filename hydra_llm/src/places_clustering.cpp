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
  field(config.color_by_task, "color_by_task");
}

PlaceClustering::PlaceClustering(const Config& config)
    : config(config::checkValid(config)),
      clustering_(config.clustering.create()),
      region_id_('l', 0) {
  CHECK(clustering_);
}

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
    attrs->semantic_feature = clusters[i]->feature;
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

void PlaceClustering::clusterPlaces(DynamicSceneGraph& graph) {
  // TODO(nathan) maintain this at the updater func level?
  const auto& places = graph.getLayer(DsgLayers::PLACES);
  NodeEmbeddingMap valid_features;
  for (auto&& [node_id, node] : places.nodes()) {
    const auto& attrs = node->attributes<SemanticNodeAttributes>();
    if (attrs.semantic_feature.size() <= 1) {
      continue;
    }

    valid_features[node_id] = &(attrs.semantic_feature);
  }

  if (valid_features.empty()) {
    VLOG(VLEVEL_TRACE) << "Need to have at least one valid place feature";
    return;
  }

  const auto clusters = clustering_->cluster(places, valid_features);
  updateGraphBatch(graph, clusters);
}

}  // namespace hydra::llm
