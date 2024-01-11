#include "hydra_llm/view_database.h"

namespace hydra::llm {

struct ClipView {
  using Ptr = std::shared_ptr<ClipView>;
  uint64_t timestamp_ns;
  Eigen::Isometry3d sensor_T_world;
  const Sensor* sensor = nullptr;
  const ClipEmbedding* clip = nullptr;
};

// TODO(nathan) virtual view selector
std::optional<NodeId> getBestView(const std::map<NodeId, ClipView>& views,
                                  const PlaceNodeAttributes& attrs) {
  double min_dist = std::numeric_limits<double>::max();
  std::optional<NodeId> best_view;
  for (auto&& [agent_id, view] : views) {
    const Eigen::Vector3f p_s = (view.sensor_T_world * attrs.position).cast<float>();
    if (!view.sensor->pointIsInViewFrustum(p_s)) {
      continue;
    }

    // heuristic to pick the view that's closest to the boundary of the free-space
    // sphere
    const auto dist = std::abs(attrs.distance - p_s.norm());
    if (dist < min_dist) {
      best_view = agent_id;
      min_dist = dist;
    }
  }

  return best_view;
}

ViewDatabase::ViewDatabase() {}

ViewDatabase::~ViewDatabase() {}

void ViewDatabase::addView(NodeId node,
                           ClipEmbedding::Ptr&& embedding,
                           const std::shared_ptr<Sensor>& sensor) {
  auto iter = entries_.find(node);
  if (iter != entries_.end()) {
    return; // required for threadsafey
  }

  entries_.emplace(node, ViewEntry{node, std::move(embedding), sensor});
}

const ViewEntry* ViewDatabase::getView(NodeId node) const {
  auto iter = entries_.find(node);
  if (iter == entries_.end()) {
    return nullptr;
  }

  return &(iter->second);
}

inline Eigen::Isometry3d getAgentPose(const AgentNodeAttributes& attrs) {
  return Eigen::Translation3d(attrs.position) * attrs.world_R_body;
}

void ViewDatabase::updateAssignments(const DynamicSceneGraph& graph,
                                     const std::vector<NodeId>& active_agents,
                                     const std::unordered_set<NodeId>& active_places,
                                     std::map<NodeId, NodeId>& best_views) const {
  std::map<NodeId, ClipView> views;
  for (const auto node_id : active_agents) {
    const auto entry = getView(node_id);
    if (!entry) {
      continue;
    }

    const auto& attrs = graph.getNode(node_id)->get().attributes<AgentNodeAttributes>();
    auto iter = views.emplace(node_id, ClipView{}).first;
    auto& view = iter->second;
    view.clip = entry->clip.get();
    view.sensor = entry->sensor.get();

    const Eigen::Isometry3d world_T_body = getAgentPose(attrs);
    view.sensor_T_world = (world_T_body * view.sensor->body_T_sensor()).inverse();
  }

  for (const auto node_id : active_places) {
    const auto& attrs = graph.getNode(node_id)->get().attributes<PlaceNodeAttributes>();
    const auto best_view = getBestView(views, attrs);
    if (!best_view) {
      continue;
    }

    best_views[node_id] = *best_view;
  }
}

}  // namespace hydra::llm
