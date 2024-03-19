#include "hydra_llm/object_update_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/dsg_types.h>
#include <hydra/utils/timing_utilities.h>
#include <khronos/common/utils/khronos_attribute_utils.h>

#include "hydra_llm/agglomerative_clustering.h"

namespace hydra::llm {

using config::VirtualConfig;
using timing::ScopedTimer;

bool isNodeActive(const SceneGraphNode& node,
                  const std::map<NodeId, size_t>& node_to_component) {
  if (node.attributes().is_active) {
    return true;
  }

  return !node_to_component.count(node.id);
}

IdTracker::IdTracker(size_t start) : idx_(start) {}

size_t IdTracker::next() const {
  // default to assuming unused is empty
  size_t new_id = idx_;
  if (unused_.empty()) {
    ++idx_;
  } else {
    new_id = unused_.front();
    unused_.pop_front();
  }

  return new_id;
}

void IdTracker::markFree(size_t idx) const {
  // validate that ID could have come from existing object
  if (idx < idx_) {
    unused_.push_back(idx);
  }
}

OverlapIntersection::OverlapIntersection(const Config& config)
    : IntersectionPolicy(), config(config) {}

bool OverlapIntersection::call(const KhronosObjectAttributes& lhs,
                               const KhronosObjectAttributes& rhs) const {
  const auto& bbox_lhs = lhs.bounding_box;
  const auto& bbox_rhs = rhs.bounding_box;
  // TODO(nathan) get expanded bounding boxes
  Eigen::Array<float, 4, 3> combined;
  combined.row(0) = bbox_lhs.min;
  combined.row(1) = bbox_rhs.min;
  combined.row(2) = bbox_lhs.max;
  combined.row(3) = bbox_rhs.max;

  const auto min = combined.colwise().minCoeff();
  combined.rowwise() -= min;
  return (combined.row(0) <= combined.row(3)).all() &&
         (combined.row(2) >= combined.row(1)).all();
}

void declare_config(OverlapIntersection::Config& config) {
  using namespace config;
  name("OverlapIntersection::Config");
  field(config.tolerance, "tolerance");
}

ComponentInfo::ComponentInfo(const IBEdgeSelector::Config& config,
                             const EdgeSelector::ScoreFunc& f_score,
                             const SceneGraphLayer& layer,
                             const std::vector<NodeId>& nodes)
    : edge_selector(config), ws(layer, nodes), segments(nodes) {
  clusterAgglomerative(ws, edge_selector, f_score);
}

ObjectUpdateFunctor::ObjectUpdateFunctor(const Config& config)
    : config(config::checkValid(config)),
      edge_checker_(config.edge_checker.create()),
      tasks_(config.tasks.create()),
      metric_(config.metric.create()),
      next_node_id_(config.prefix, 0) {}

MergeMap ObjectUpdateFunctor::call(SharedDsgInfo& dsg, const UpdateInfo& info) const {
  ScopedTimer timer("backend/object_clustering", info.timestamp_ns);
  auto& graph = *dsg.graph;
  // detect edges between segments (and active connected components)
  const auto active_components = addSegmentEdges(graph);
  // remove all previous components that are active
  clearActiveComponents(graph, active_components);
  // construct new components and cluster into objects
  detectObjects(graph);
  // we never have explict merges (clustering takes care of them)
  return {};
}

void ObjectUpdateFunctor::clearActiveComponents(DynamicSceneGraph& graph,
                                                const std::set<size_t>& active) const {
  auto iter = components_.begin();
  while (iter != components_.end()) {
    if (!active.count(iter->first)) {
      ++iter;
      continue;
    }

    for (const auto node_id : iter->second->segments) {
      node_to_component_.erase(node_id);
    }

    for (const auto& node_id : iter->second->objects) {
      graph.removeNode(node_id);
    }

    components_ids_.markFree(iter->first);
    iter = components_.erase(iter);
  }
}

std::set<size_t> ObjectUpdateFunctor::addSegmentEdges(DynamicSceneGraph& graph) const {
  const auto& segments = graph.getLayer(DsgLayers::SEGMENTS);

  std::set<size_t> active_components;
  for (auto&& [node_id, node] : segments.nodes()) {
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    if (!attrs.is_active) {
      continue;
    }

    const auto iter = node_to_component_.find(node_id);
    if (iter != node_to_component_.end()) {
      active_components.insert(iter->second);
      continue;
    }

    // TODO(nathan) do something smarter than pairwise iteration
    for (auto&& [other_id, other_node] : segments.nodes()) {
      if (other_id == node_id) {
        continue;
      }

      const auto& other_attrs = other_node->attributes<KhronosObjectAttributes>();
      if (edge_checker_->call(attrs, other_attrs)) {
        graph.insertEdge(node_id, other_id);
        const auto oiter = node_to_component_.find(other_id);
        if (oiter != node_to_component_.end()) {
          active_components.insert(oiter->second);
        }
      }
    }
  }

  return active_components;
}

void ObjectUpdateFunctor::detectObjects(DynamicSceneGraph& graph) const {
  const auto& segments = graph.getLayer(DsgLayers::SEGMENTS);

  // connected component search
  const auto new_components = graph_utilities::getConnectedComponents(
      segments,
      [&](const auto& n) { return isNodeActive(n, node_to_component_); },
      [&](const auto& edge) {
        const auto source_active =
            isNodeActive(segments.getNode(edge.source).value(), node_to_component_);
        const auto target_active =
            isNodeActive(segments.getNode(edge.target).value(), node_to_component_);
        return source_active && target_active;
      });

  const auto f_score = [this](const Eigen::VectorXd& x) {
    return tasks_->getBestScore(*metric_, x).score;
  };

  // reassign components
  for (const auto& nodes : new_components) {
    size_t new_id = components_ids_.next();
    auto new_component =
        std::make_unique<ComponentInfo>(config.selector, f_score, segments, nodes);

    const auto clusters = new_component->ws.getClusters();
    for (const auto& cluster : clusters) {
      std::set<NodeId> parents;
      auto iter = cluster.begin();
      const auto& node = graph.getNode(*iter)->get();
      const auto parent = node.getParent();
      auto attrs_ptr = node.attributes().clone();
      ++iter;

      auto& attrs =
          *CHECK_NOTNULL(dynamic_cast<KhronosObjectAttributes*>(attrs_ptr.get()));
      while (iter != cluster.end()) {
        const auto& other = graph.getNode(*iter)->get();
        khronos::mergeObjectAttributes(other.attributes<KhronosObjectAttributes>(),
                                       attrs);
      }

      graph.emplaceNode(DsgLayers::OBJECTS, next_node_id_, std::move(attrs_ptr));
      new_component->objects.push_back(next_node_id_);

      if (parent) {
        graph.insertEdge(next_node_id_, *parent);
      } else {
        LOG(WARNING) << "Inserted clustered object '" << next_node_id_.getLabel()
                     << "' without parent!";
      }

      ++next_node_id_;
    }

    components_.emplace(new_id, std::move(new_component));
  }
}

void declare_config(ObjectUpdateFunctor::Config& config) {
  using namespace config;
  name("ObjectUpdateFunctor::Config");
  field<CharConversion>(config.prefix, "prefix");
  field(config.edge_checker, "edge_checker");
  field(config.tasks, "tasks");
  field(config.metric, "metric");
  field(config.selector, "selector");
}

}  // namespace hydra::llm
