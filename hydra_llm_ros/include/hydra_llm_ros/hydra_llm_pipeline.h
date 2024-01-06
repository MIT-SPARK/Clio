#pragma once
#include <hydra_ros/common/hydra_ros_pipeline.h>

namespace hydra::llm {

class HydraLLMPipeline : public HydraRosPipeline {
 public:
  HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id);

  virtual ~HydraLLMPipeline();
};

}  // namespace hydra::llm
