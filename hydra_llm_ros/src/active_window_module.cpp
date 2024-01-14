#include "hydra_llm_ros/active_window_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>

namespace hydra::llm {

void declare_config(ActiveWindowModule::Config& config) {
  using namespace config;
  name("ActiveWindowModule::Config");
  base<ReconstructionConfig>(config);
}

ActiveWindowModule::ActiveWindowModule(const Config& config,
                                       const OutputQueue::Ptr& output_queue)
    : ReconstructionModule(config, output_queue), config(config::checkValid(config)) {}

ActiveWindowModule::~ActiveWindowModule() {}

void ActiveWindowModule::start() { ReconstructionModule::start(); }

void ActiveWindowModule::stop() { ReconstructionModule::stop(); }

void ActiveWindowModule::save(const LogSetup& log_setup) {
  ReconstructionModule::save(log_setup);
}

std::string ActiveWindowModule::printInfo() const {
  std::stringstream ss;
  ss << std::endl << config::toString(config);
  return ss.str();
}

}  // namespace hydra::llm
