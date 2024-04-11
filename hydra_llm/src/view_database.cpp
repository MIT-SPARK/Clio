#include "hydra_llm/view_database.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <glog/logging.h>

namespace hydra::llm {

void declare_config(ViewDatabase::Config& config) {
  using namespace config;
  name("ViewDatabase::Config");
  field(config.view_selection_method, "view_selection_method");
}

using NodeSet = std::unordered_set<NodeId>;

struct BoundaryViewSelector : ViewSelector {
  void selectFeature(const std::map<NodeId, ClipView>& views,
                     PlaceNodeAttributes& attrs) const override {
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

    if (best_view) {
      attrs.semantic_feature = *views.at(*best_view).feature;
    }
  }

  inline static const auto registration_ =
      config::Registration<ViewSelector, BoundaryViewSelector>("boundary");
};

struct ClosestViewSelector : ViewSelector {
  void selectFeature(const std::map<NodeId, ClipView>& views,
                     PlaceNodeAttributes& attrs) const override {
    double min_dist = std::numeric_limits<double>::max();
    std::optional<NodeId> best_view;
    for (auto&& [agent_id, view] : views) {
      const Eigen::Vector3f p_s = (view.sensor_T_world * attrs.position).cast<float>();
      if (!view.sensor->pointIsInViewFrustum(p_s)) {
        continue;
      }

      // norm of position in sensor frame is distance in world frame
      const auto dist = p_s.norm();
      if (dist < min_dist) {
        best_view = agent_id;
        min_dist = dist;
      }
    }

    if (best_view) {
      attrs.semantic_feature = *views.at(*best_view).feature;
    }
  }

  inline static const auto registration_ =
      config::Registration<ViewSelector, ClosestViewSelector>("closest");
};

struct FusionViewSelector : ViewSelector {
  void selectFeature(const std::map<NodeId, ClipView>& views,
                     PlaceNodeAttributes& attrs) const override {
    size_t num_visible = 0;
    for (auto&& [agent_id, view] : views) {
      const Eigen::Vector3f p_s = (view.sensor_T_world * attrs.position).cast<float>();
      if (!view.sensor->pointIsInViewFrustum(p_s)) {
        continue;
      }

      if (!num_visible) {
        attrs.semantic_feature = *view.feature;
      } else {
        attrs.semantic_feature += *view.feature;
      }
      ++num_visible;
    }

    if (num_visible > 0) {
      attrs.semantic_feature /= num_visible;
    }
  }

  inline static const auto registration_ =
      config::Registration<ViewSelector, FusionViewSelector>("fusion");
};

ViewDatabase::ViewDatabase(const Config& config)
    : view_selector_(config::create<ViewSelector>(config.view_selection_method)) {
  CHECK(view_selector_);
}

ViewDatabase::~ViewDatabase() {}

void ViewDatabase::addView(NodeId node,
                           const Eigen::VectorXd& feature,
                           const Sensor* sensor) {
  auto iter = entries_.find(node);
  if (iter != entries_.end()) {
    return;  // required for threadsafey
  }

  entries_.emplace(node, ViewEntry{node, feature, sensor});
  active_views_.insert(node);
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
                                     const NodeSet& active_places) const {
  std::map<NodeId, ClipView> views;
  auto iter = active_views_.begin();
  while (iter != active_views_.end()) {
    const auto node_id = *iter;
    const auto entry = getView(node_id);
    if (!entry) {
      continue;
    }

    const auto& attrs = graph.getNode(node_id)->get().attributes<AgentNodeAttributes>();
    const Eigen::Isometry3d world_T_body = getAgentPose(attrs);
    const Eigen::Isometry3d sensor_T_world =
        (world_T_body * entry->sensor->body_T_sensor()).inverse();
    bool visible = false;
    for (const auto node_id : active_places) {
      const Eigen::Vector3f p_s =
          (sensor_T_world * graph.getPosition(node_id)).cast<float>();
      if (entry->sensor->pointIsInViewFrustum(p_s)) {
        visible = true;
        break;
      }
    }

    if (!visible) {
      iter = active_views_.erase(iter);
      continue;
    }

    views.emplace(node_id, ClipView{sensor_T_world, entry->sensor, &entry->feature});
    ++iter;
  }

  VLOG(2) << "Assigning features with " << views.size() << " active view(s)";
  for (const auto node_id : active_places) {
    auto& attrs = graph.getNode(node_id)->get().attributes<PlaceNodeAttributes>();
    view_selector_->selectFeature(views, attrs);
  }
}

}  // namespace hydra::llm
