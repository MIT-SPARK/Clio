#include "hydra_llm_ros/ros_task_embeddings.h"

#include <config_utilities/config.h>

namespace hydra::llm {

void declare_config(RosTaskEmbeddings::Config&) {
  using namespace config;
  name("RosTaskEmbeddings::Config");
}

RosTaskEmbeddings::RosTaskEmbeddings(const Config& config) {}

}  // namespace hydra::llm
