#include "clio/region_update_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/rooms/room_utilities.h>
#include <hydra/utils/display_utilities.h>
#include <hydra/utils/timing_utilities.h>

namespace clio {
namespace {

static const auto functor_reg =
    config::RegistrationWithConfig<hydra::UpdateFunctor,
                                   RegionUpdateFunctor,
                                   RegionUpdateFunctor::Config>("RegionsIBFunctor");

}  // namespace

using hydra::MergeList;
using hydra::timing::ScopedTimer;
using namespace spark_dsg;

using Clusters = std::vector<Cluster::Ptr>;

void declare_config(RegionUpdateFunctor::Config& config) {
  using namespace config;
  name("RegionUpdateFunctorConfig::Config");
  field(config.clustering, "clustering");
}

RegionUpdateFunctor::RegionUpdateFunctor(const Config& config)
    : config(config::checkValid(config)),
      region_id_('l', 0),
      clustering_(config.clustering) {}

MergeList RegionUpdateFunctor::call(const DynamicSceneGraph&,
                                    hydra::SharedDsgInfo& dsg,
                                    const hydra::UpdateInfo::ConstPtr& info) const {
  ScopedTimer timer("backend/region_clustering", info->timestamp_ns);

  // TODO(nathan) cache this computation
  const auto& places = dsg.graph->getLayer(DsgLayers::PLACES);
  AgglomerativeClustering::NodeEmbeddingMap valid_features;
  for (auto&& [node_id, node] : places.nodes()) {
    const auto& attrs = node->attributes<SemanticNodeAttributes>();
    if (attrs.semantic_feature.size() <= 1) {
      continue;
    }

    valid_features[node_id] = attrs.semantic_feature.rightCols<1>();
  }

  if (valid_features.empty()) {
    VLOG(2) << "Need to have at least one valid place feature";
    return {};
  }

  const auto clusters = clustering_.cluster(places, valid_features);
  updateGraphBatch(*dsg.graph, clusters);
  return {};
}

void RegionUpdateFunctor::updateGraphBatch(DynamicSceneGraph& graph,
                                           const Clusters& clusters) const {
  VLOG(2) << "Got " << clusters.size() << " cluster(s)";

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
    auto attrs = std::make_unique<SemanticNodeAttributes>();
    attrs->semantic_label = 0;
    attrs->name = clusters[i]->best_task_name;
    attrs->semantic_feature = clusters[i]->feature;
    attrs->semantic_label = clusters[i]->best_task_index;
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
    node->attributes().position = hydra::getRoomPosition(places, to_use);
  }

  hydra::addEdgesToRoomLayer(graph, new_nodes);
}

}  // namespace clio
