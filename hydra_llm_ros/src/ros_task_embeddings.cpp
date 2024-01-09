#include "hydra_llm_ros/ros_task_embeddings.h"

#include <config_utilities/config.h>
#include <glog/logging.h>
#include <llm/RequestTaskEmbeddings.h>
#include <ros/ros.h>

namespace hydra::llm {

void declare_config(RosTaskEmbeddings::Config&) {
  using namespace config;
  name("RosTaskEmbeddings::Config");
}

RosTaskEmbeddings::RosTaskEmbeddings(const Config&) {
  LOG(INFO) << "Waiting for task service on '/get_tasks'";
  ros::service::waitForService("/get_tasks");
  ::llm::RequestTaskEmbeddings msg;
  CHECK(ros::service::call("/get_tasks", msg));

  for (const auto& embedding : msg.response.embeddings) {
    const auto& vec = embedding.elements;
    embeddings.emplace_back(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
  }

  tasks = msg.response.tasks;
}

}  // namespace hydra::llm
