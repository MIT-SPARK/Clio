#include "hydra_llm/places_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

namespace hydra::llm {

void declare_config(PlaceClustering::Config& config) {
  using namespace config;
  name("PlaceClustering::Config");
  field(config.similarity_threshold, "similarity_threshold");
  field(config.run_preprune, "run_preprune");
}

const ClipView* getBestView(const std::map<size_t, ClipView::Ptr>& views,
                            const PlaceNodeAttributes& attrs) {
  double min_dist = std::numeric_limits<double>::max();
  const ClipView* best_view = nullptr;
  for (const auto& id_view_pair : views) {
    const auto& view = id_view_pair.second;
    const auto& sensor = *(view->sensor);

    const Eigen::Vector3f p_s = (view->world_T_sensor * attrs.position).cast<float>();
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
    : config(config::checkValid(config)) {}

PlaceClustering::~PlaceClustering() {}

void PlaceClustering::clusterPlaces(DynamicSceneGraph& graph,
                                    const std::map<size_t, ClipView::Ptr>& views,
                                    const std::unordered_set<NodeId>& nodes) {
  if (views.empty()) {
    LOG(ERROR) << "Need to have at least one view";
    return;
  }

  std::map<NodeId, const ClipView*> assigned_views;
  const auto& places = graph.getLayer(DsgLayers::PLACES);
  for (const auto node_id : nodes) {
    const auto& attrs =
        places.getNode(node_id)->get().attributes<PlaceNodeAttributes>();
    assigned_views[node_id] = getBestView(views, attrs);
  }
}

}  // namespace hydra::llm
