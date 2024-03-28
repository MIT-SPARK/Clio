#pragma once
#include <hydra_llm/region_update_functor.h>
#include <hydra_llm/object_update_functor.h>
#include <hydra_ros/common/hydra_ros_pipeline.h>

namespace hydra::llm {

class HydraLLMPipeline : public HydraRosPipeline {
 public:
  HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id);

  virtual ~HydraLLMPipeline();

  void init() override;

  void stop() override;

 protected:
  void initReconstruction() override;

  void configureClustering();

  std::shared_ptr<ObjectUpdateFunctor> object_clustering_;
  std::shared_ptr<RegionUpdateFunctor> region_clustering_;
};

}  // namespace hydra::llm
