#pragma once
#include <hydra/common/dsg_types.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

namespace hydra::llm {

using NodeColor = SemanticNodeAttributes::ColorVector;

class NodeFilter {
 public:
  using Ptr = std::shared_ptr<NodeFilter>;
  using Func = std::function<bool(const SceneGraphNode& node)>;

  struct Config {
    LayerId layer = DsgLayers::PLACES;
    LayerId child_layer = DsgLayers::OBJECTS;
  } const config;

  NodeFilter(const Config& config, const ros::NodeHandle& nh);

  virtual ~NodeFilter() = default;

  bool hasChange() const { return has_change_; }

  void clearChangeFlag() { has_change_ = false; }

  Func getFilter() const;

 private:
  void handleFilter(const std_msgs::String& msg);

  ros::NodeHandle nh_;
  ros::Subscriber sub_;

  bool has_change_;
  std::set<NodeId> filter_;
};

}  // namespace hydra::llm
