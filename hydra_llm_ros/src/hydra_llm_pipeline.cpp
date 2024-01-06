#include "hydra_llm_ros/hydra_llm_pipeline.h"

namespace hydra::llm {

HydraLLMPipeline::HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraRosPipeline(nh, robot_id) {}

HydraLLMPipeline::~HydraLLMPipeline() {}

}  // namespace hydra::llm
