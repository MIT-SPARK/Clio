#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/module.h>
#include <hydra/reconstruction/sensor.h>
#include <khronos_ros/input/input_synchronizer.h>
#include <ros/ros.h>

namespace hydra::llm {

struct InputSensorConfig {
  std::string ns = "input";
  config::VirtualConfig<Sensor> sensor;
};

void declare_config(InputSensorConfig& config);

class KhronosInputModule : public Module {
 public:
  using InputQueue = khronos::InputSynchronizer::InputQueue;

  KhronosInputModule(const ros::NodeHandle& nh,
                     const std::vector<InputSensorConfig>& inputs,
                     const InputQueue::Ptr& data_queue);

  virtual ~KhronosInputModule() = default;

  void start() override;

  void stop() override;

  void save(const LogSetup&) override;

  std::string printInfo() const override;

  std::vector<std::shared_ptr<khronos::InputSynchronizer>> inputs_;
};

}  // namespace hydra::llm
