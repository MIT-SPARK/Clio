#pragma once
#include <hydra/common/dsg_types.h>
#include <hydra/reconstruction/sensor.h>

#include "hydra_llm/clip_types.h"

namespace hydra::llm {

struct ViewEntry {
  NodeId node_id;
  const Eigen::VectorXd feature;
  const Sensor* sensor;
};

struct ClipView {
  using Ptr = std::shared_ptr<ClipView>;
  Eigen::Isometry3d sensor_T_world;
  const Sensor* sensor = nullptr;
  const Eigen::VectorXd* feature = nullptr;
};

struct ViewSelector {
  virtual void selectFeature(const std::map<NodeId, ClipView>& views,
                             PlaceNodeAttributes& attrs) const = 0;
};

class ViewDatabase {
 public:
  using Ptr = std::shared_ptr<ViewDatabase>;

  struct Config {
    std::string view_selection_method = "boundary";
  };

  explicit ViewDatabase(const Config& config);

  ~ViewDatabase();

  void addView(NodeId node, const Eigen::VectorXd& feature, const Sensor* sensor);

  const ViewEntry* getView(NodeId node) const;

  void updateAssignments(const DynamicSceneGraph& graph,
                         const std::unordered_set<NodeId>& active_places) const;

 protected:
  mutable std::set<NodeId> active_views_;
  std::map<NodeId, ViewEntry> entries_;
  std::unique_ptr<ViewSelector> view_selector_;
};

void declare_config(ViewDatabase::Config& config);

}  // namespace hydra::llm
