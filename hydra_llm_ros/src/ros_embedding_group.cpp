#include "hydra_llm_ros/ros_embedding_group.h"

#include <config_utilities/config.h>
#include <glog/logging.h>
#include <llm/RequestTaskEmbeddings.h>
#include <ros/ros.h>

namespace hydra::llm {

void declare_config(RosEmbeddingGroup::Config& config) {
  using namespace config;
  name("RosEmbeddingGroup::Config");
  field(config.service_name, "service_name");
}

RosEmbeddingGroup::RosEmbeddingGroup(const Config& config) {
  LOG(INFO) << "Waiting for task service on '/get_tasks'";
  ros::service::waitForService(config.service_name);
  ::llm::RequestTaskEmbeddings msg;
  CHECK(ros::service::call(config.service_name, msg));

  for (const auto& embedding : msg.response.embeddings) {
    const auto& vec = embedding.elements;
    embeddings.emplace_back(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
  }

  tasks = msg.response.tasks;
}

}  // namespace hydra::llm
