#pragma once
#include <hydra/common/dsg_types.h>

#include "hydra_llm/clip_types.h"

namespace hydra::llm {

struct ViewEntry {
  NodeId node_id;
  ClipEmbedding::Ptr clip;
  std::shared_ptr<Sensor> sensor;
};

class ViewDatabase {
 public:
  using Ptr = std::shared_ptr<ViewDatabase>;

  explicit ViewDatabase();

  ~ViewDatabase();

  void addView(NodeId node, ClipEmbedding::Ptr&& embedding);

  const ViewEntry* getView(NodeId node) const;

  void updateAssignments(const DynamicSceneGraph& graph,
                         const std::vector<NodeId>& active_agents,
                         const std::unordered_set<NodeId>& active_places,
                         std::map<NodeId, NodeId>& best_views) const;

 protected:
  std::map<NodeId, ViewEntry> entries_;
};

}  // namespace hydra::llm
