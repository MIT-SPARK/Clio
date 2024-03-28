#include "hydra_llm_ros/ros_embedding_group.h"

#include <config_utilities/config.h>
#include <glog/logging.h>
#include <llm/RequestEmbedding.h>
#include <llm/RequestTaskEmbeddings.h>
#include <ros/ros.h>

namespace hydra::llm {

void declare_config(RosEmbeddingGroup::Config& config) {
  using namespace config;
  name("RosEmbeddingGroup::Config");
  field(config.service_ns, "service_ns");
  field(config.silent_wait, "silent_wait");
  field(config.prompts, "prompts");
}

RosEmbeddingGroup::RosEmbeddingGroup(const Config& config) {
  const std::string base_name =
      config.prompts.empty() ? "/get_tasks" : "/get_embedding";
  const std::string service_name = ros::names::append(config.service_ns, base_name);

  LOG_IF(INFO, !config.silent_wait)
      << "Waiting for task service on '" << ros::names::resolve(service_name) << "'";
  ros::service::waitForService(service_name);

  if (config.prompts.empty()) {
    ::llm::RequestTaskEmbeddings srv;
    CHECK(ros::service::call(service_name, srv));
    const auto& resp = srv.response;
    VLOG(5) << "Got " << resp.tasks.size() << " tasks:";
    for (size_t i = 0; i < resp.embeddings.size(); ++i) {
      VLOG(5) << " - " << i << ": " << resp.tasks[i];
      tasks.push_back(resp.tasks[i]);
      const auto& vec = resp.embeddings[i].elements;
      embeddings.emplace_back(
          Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
    }
  } else {
    for (const auto& prompt : config.prompts) {
      ::llm::RequestEmbedding msg;
      msg.request.prompt = prompt;
      CHECK(ros::service::call(service_name, msg));
      tasks.push_back(prompt);
      const auto& vec = msg.response.embedding.elements;
      embeddings.emplace_back(
          Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size()));
    }
  }
}

}  // namespace hydra::llm
