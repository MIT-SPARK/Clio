#include "hydra_llm_ros/node_filter.h"

#include <config_utilities/config.h>
#include <glog/logging.h>

namespace hydra::llm {

void declare_config(NodeFilter::Config& config) {
  using namespace config;
  name("NodeFilter::Config");
  field(config.layer, "layer");
  field(config.child_layer, "child_layer");
}

NodeFilter::NodeFilter(const Config& config, const ros::NodeHandle& nh)
    : config(config), nh_(nh), has_change_(false) {
  sub_ = nh_.subscribe("node_filter", 1, &NodeFilter::handleFilter, this);
}

NodeFilter::Func NodeFilter::getFilter() const {
  if (filter_.empty()) {
    return {};
  }

  return [this](const SceneGraphNode& node) -> bool {
    if (node.layer == config.layer) {
      return filter_.count(node.id);
    }

    const auto parent = node.getParent();
    if (!parent) {
      return false;
    }

    return filter_.count(*parent);
  };
}

void NodeFilter::handleFilter(const std_msgs::String& msg) {
  filter_.clear();

  std::string curr_token;
  std::stringstream ss(msg.data);
  while (std::getline(ss, curr_token, ',')) {
    const auto lparen = curr_token.find("(");
    if (lparen == std::string::npos) {
      LOG(WARNING) << "Invalid node ID: " << curr_token;
      continue;
    }

    const auto rparen = curr_token.find(")");
    if (rparen == std::string::npos) {
      LOG(WARNING) << "Invalid node ID: " << curr_token;
      continue;
    }

    if (rparen == 0) {
      LOG(WARNING) << "Invalid node ID: " << curr_token;
      continue;
    }

    const auto size = rparen - lparen - 1;
    const auto index_str = curr_token.substr(lparen + 1, size);
    const auto index = std::stoull(index_str);
    const auto prefix = curr_token[lparen - 1];
    NodeSymbol node_id(prefix, index);
    filter_.insert(node_id);
  }

  has_change_ = true;
}

}  // namespace hydra::llm
