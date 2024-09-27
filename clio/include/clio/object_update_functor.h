#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/backend/update_functions.h>
#include <hydra/utils/id_tracker.h>

#include <map>

#include "clio/ib_edge_selector.h"

namespace clio {

struct IntersectionPolicy {
  using Ptr = std::unique_ptr<IntersectionPolicy>;
  virtual ~IntersectionPolicy() = default;
  virtual bool call(const spark_dsg::KhronosObjectAttributes& lhs,
                    const spark_dsg::KhronosObjectAttributes& rhs) const = 0;
  bool operator()(const spark_dsg::KhronosObjectAttributes& lhs,
                  const spark_dsg::KhronosObjectAttributes& rhs) const {
    return call(lhs, rhs);
  }
};

struct OverlapIntersection : public IntersectionPolicy {
  struct Config {
    double tolerance = 0.1;
  } const config;

  explicit OverlapIntersection(const Config& config);

  bool call(const spark_dsg::KhronosObjectAttributes& lhs,
            const spark_dsg::KhronosObjectAttributes& rhs) const;
};

void declare_config(OverlapIntersection::Config& config);

struct ComponentInfo {
  using Ptr = std::unique_ptr<ComponentInfo>;

  ComponentInfo(const IBEdgeSelector::Config& config,
                const hydra::EmbeddingGroup& tasks,
                const hydra::EmbeddingDistance& metric,
                const spark_dsg::SceneGraphLayer& segments,
                const std::vector<NodeId>& nodes,
                double I_xy_full);

  IBEdgeSelector edge_selector;
  ClusteringWorkspace ws;

  std::vector<NodeId> segments;
  std::vector<NodeId> objects;
};

class ObjectUpdateFunctor : public hydra::UpdateFunctor {
 public:
  struct Config {
    char prefix = 'O';
    config::VirtualConfig<IntersectionPolicy> edge_checker{
        OverlapIntersection::Config()};
    config::VirtualConfig<hydra::EmbeddingGroup> tasks;
    config::VirtualConfig<hydra::EmbeddingDistance> metric{
        hydra::CosineDistance::Config()};
    IBEdgeSelector::Config selector;
    double min_segment_score = 0.2;
    double min_object_score = 0.2;
    double neighbor_max_distance = 0.0;
  } const config;

  explicit ObjectUpdateFunctor(const Config& config);

  hydra::MergeList call(const spark_dsg::DynamicSceneGraph& unmerged,
                        hydra::SharedDsgInfo& dsg,
                        const hydra::UpdateInfo::ConstPtr& info) const override;

  std::set<size_t> addSegmentEdges(spark_dsg::DynamicSceneGraph& graph) const;

  void clearActiveComponents(spark_dsg::DynamicSceneGraph& graph,
                             const std::set<size_t>& active) const;

  void detectObjects(spark_dsg::DynamicSceneGraph& segments) const;

  void updateActiveParents(spark_dsg::DynamicSceneGraph& graph) const;

 protected:
  IntersectionPolicy::Ptr edge_checker_;
  hydra::EmbeddingGroup::Ptr tasks_;
  std::unique_ptr<hydra::EmbeddingDistance> metric_;

  hydra::IdTracker components_ids_;
  mutable std::set<NodeId> ignored_;
  mutable std::set<NodeId> active_;
  mutable NodeSymbol next_node_id_;
  mutable std::map<size_t, ComponentInfo::Ptr> components_;
  mutable std::map<NodeId, size_t> node_to_component_;
};

void declare_config(ObjectUpdateFunctor::Config& config);

}  // namespace clio
