#include "hydra_llm/view_database.h"

namespace hydra::llm {

ViewDatabase::ViewDatabase() {}

ViewDatabase::~ViewDatabase() {}

void ViewDatabase::addView(NodeId node, ClipEmbedding::Ptr&& embedding) {
  // TODO(nathan) threadsafety
  auto iter = entries_.find(node);
  if (iter == entries_.end()) {
    iter = entries_.emplace(node, ViewEntry{node, nullptr}).first;
  }

  iter->second.clip = std::move(embedding);
}

const ViewEntry* ViewDatabase::getView(NodeId node) const {
  // TODO(nathan) threadsafety
  auto iter = entries_.find(node);
  if (iter == entries_.end()) {
    return nullptr;
  }

  return &(iter->second);
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

/*  std::map<NodeId, const ClipEmbedding*> assigned_views;*/
  /*const auto& places = graph.getLayer(DsgLayers::PLACES);*/
  /*std::map<NodeId, std::set<NodeId>> region_sets;*/
  /*for (const auto node_id : nodes) {*/
    /*const SceneGraphNode& node = places.getNode(node_id)->get();*/
    /*const auto& attrs = node.attributes<PlaceNodeAttributes>();*/
    /*const auto best_view = getBestView(views, attrs);*/
    /*if (!best_view) {*/
      /*continue;*/
    /*}*/

    /*assigned_views[node_id] = best_view->clip.get();*/
    /*const auto parent = node.getParent();*/
    /*if (parent) {*/
      /*auto iter = region_sets.find(*parent);*/
      /*if (iter == region_sets.end()) {*/
        /*iter = region_sets.emplace(*parent, std::set<NodeId>()).first;*/
      /*}*/

      /*iter->second.insert(node_id);*/
    /*}*/
  /*}*/

}  // namespace hydra::llm
