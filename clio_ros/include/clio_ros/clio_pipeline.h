#pragma once
#include <hydra_ros/hydra_ros_pipeline.h>

namespace clio {

class ClioPipeline : public hydra::HydraRosPipeline {
 public:
  ClioPipeline(const ros::NodeHandle& nh, int robot_id);

  virtual ~ClioPipeline();

  void init() override;

  void stop() override;

 protected:
  void initReconstruction() override;
};

}  // namespace clio
