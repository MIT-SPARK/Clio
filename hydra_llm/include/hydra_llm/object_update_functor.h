#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/backend/update_functions.h>
#include <hydra/utils/disjoint_set.h>

#include <map>

#include "hydra_llm/embedding_group.h"
#include "hydra_llm/ib_edge_selector.h"

namespace hydra::llm {

class IdTracker {
 public:
  explicit IdTracker(size_t start = 0);

  size_t next() const;

  void markFree(size_t idx) const;

 private:
  mutable size_t idx_;
  mutable std::list<size_t> unused_;
};

struct IntersectionPolicy {
  using Ptr = std::unique_ptr<IntersectionPolicy>;
  virtual bool call(const KhronosObjectAttributes& lhs,
                    const KhronosObjectAttributes& rhs) const = 0;
  bool operator()(const KhronosObjectAttributes& lhs,
                  const KhronosObjectAttributes& rhs) const {
    return call(lhs, rhs);
  }
};

struct OverlapIntersection : public IntersectionPolicy {
  struct Config {
    double tolerance = 0.1;
  } const config;

  explicit OverlapIntersection(const Config& config);

  bool call(const KhronosObjectAttributes& lhs,
            const KhronosObjectAttributes& rhs) const;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<IntersectionPolicy, OverlapIntersection, Config>(
          "OverlapIntersection");
};

void declare_config(OverlapIntersection::Config& config);

struct ComponentInfo {
  using Ptr = std::unique_ptr<ComponentInfo>;

  ComponentInfo(const IBEdgeSelector::Config& config,
                const EdgeSelector::ScoreFunc& f_score,
                const SceneGraphLayer& segments,
                const std::vector<NodeId>& nodes);

  IBEdgeSelector edge_selector;
  ClusteringWorkspace ws;

  std::vector<NodeId> segments;
  std::vector<NodeId> objects;
};

class ObjectUpdateFunctor : dsg_updates::UpdateFunctor {
 public:
  struct Config {
    char prefix = 'O';
    config::VirtualConfig<IntersectionPolicy> edge_checker;
    config::VirtualConfig<EmbeddingGroup> tasks;
    config::VirtualConfig<EmbeddingDistance> metric;
    IBEdgeSelector::Config selector;
  } const config;

  explicit ObjectUpdateFunctor(const Config& config);

  MergeMap call(SharedDsgInfo& dsg, const UpdateInfo& info) const override;

 protected:
  std::set<size_t> addSegmentEdges(DynamicSceneGraph& graph) const;

  void clearActiveComponents(DynamicSceneGraph& graph,
                             const std::set<size_t>& active) const;

  void detectObjects(DynamicSceneGraph& segments) const;

 protected:
  IntersectionPolicy::Ptr edge_checker_;
  EmbeddingGroup::Ptr tasks_;
  std::unique_ptr<EmbeddingDistance> metric_;

  IdTracker components_ids_;
  mutable NodeSymbol next_node_id_;
  mutable std::map<size_t, ComponentInfo::Ptr> components_;
  mutable std::map<NodeId, size_t> node_to_component_;
};

void declare_config(ObjectUpdateFunctor::Config& config);

}  // namespace hydra::llm
