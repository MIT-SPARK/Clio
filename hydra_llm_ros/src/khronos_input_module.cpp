#include "hydra_llm_ros/khronos_input_module.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <hydra/common/hydra_config.h>
#include <khronos/common/utils/globals.h>

namespace hydra::llm {

void declare_config(InputSensorConfig& config) {
  using namespace config;
  name("InputSensorConfig");
  field(config.ns, "ns");
  field(config.sensor, "sensor");
}

KhronosInputModule::KhronosInputModule(const ros::NodeHandle& nh,
                                       const std::vector<InputSensorConfig>& inputs,
                                       const InputQueue::Ptr& data_queue) {
  for (const auto& conf : inputs) {
    std::shared_ptr<Sensor> sensor(config::checkValid(conf).sensor.create());
    const size_t index = khronos::Globals::addSensor(sensor);
    inputs_.push_back(std::make_shared<khronos::InputSynchronizer>(
        ros::NodeHandle(nh, conf.ns), data_queue, index));
  }
}

void KhronosInputModule::start() {
  for (const auto& input : inputs_) {
    input->start();
  }
}

void KhronosInputModule::stop() {
  for (const auto& input : inputs_) {
    input->stop();
  }
}

void KhronosInputModule::save(const LogSetup&) {}

std::string KhronosInputModule::printInfo() const {
  std::stringstream ss;
  size_t index = 0;
  for (const auto& input : inputs_) {
    ss << "input " << index << ": " << std::endl << config::toString(input->config());
    ++index;
  }
  return ss.str();
}

}  // namespace hydra::llm
