#pragma once
#include <hydra_llm/region_update_functor.h>
#include <hydra_ros/common/hydra_ros_pipeline.h>

namespace hydra::llm {

class HydraLLMPipeline : public HydraRosPipeline {
 public:
  HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id);

  virtual ~HydraLLMPipeline();

  void init() override;

 protected:
  std::shared_ptr<RegionUpdateFunctor> region_clustering_;
};

}  // namespace hydra::llm
