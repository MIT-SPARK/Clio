#include "hydra_llm_ros/active_window_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/common/common.h>

namespace hydra::llm {

void declare_config(ActiveWindowModule::Config& config) {
  using namespace config;
  name("ActiveWindowModule::Config");
  field(config.active_window, "active_window");
  field(config.max_queue_size, "max_queue_size");
  field(config.use_visualizer, "use_visualizer");
  field(config.active_window_visualizer_ns, "~active_window_viz");
}

ActiveWindowModule::ActiveWindowModule(const Config& config,
                                       const OutputQueue::Ptr& output_queue)
    : config(config::checkValid(config)),
      queue_(std::make_shared<DataInputQueue>(config.max_queue_size)),
      output_queue_(output_queue) {
  if (config.use_visualizer) {
    visualizer_ = std::make_unique<khronos::ActiveWindowVisualizer>(
        ros::NodeHandle(config.active_window_visualizer_ns));
  }
}

ActiveWindowModule::~ActiveWindowModule() {}

void ActiveWindowModule::start() {
  if (!active_window_) {
    active_window_ = std::make_unique<khronos::ActiveWindow>(config.active_window);
  }

  spin_thread_ = std::make_unique<std::thread>(&ActiveWindowModule::spin, this);
  LOG(INFO) << "[Hydra-LLM ActiveWindow] started!";
}

void ActiveWindowModule::stop() {
  should_shutdown_ = true;
  if (spin_thread_) {
    VLOG(VLEVEL_TRACE) << "[Hydra-LLM ActiveWindow] stopping active window!";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(VLEVEL_TRACE) << "[Hydra-LLM ActiveWindow] stopped!";
  }
}

void ActiveWindowModule::save(const LogSetup&) {}

std::string ActiveWindowModule::printInfo() const {
  std::stringstream ss;
  ss << std::endl << config::toString(config);
  return ss.str();
}

void ActiveWindowModule::spin() {
  while (!should_shutdown_) {
    bool has_data = queue_->poll();
    if (!has_data) {
      continue;
    }

    active_window_->processInput(*queue_->front());
    queue_->pop();

    auto output = active_window_->getOutput();
    output_queue_->push(output);

    if (visualizer_) {
      const auto frame_data =
          dynamic_cast<const khronos::FrameData*>(output->sensor_data.get());
      CHECK(frame_data);
      visualizer_->visualizeAll(
          active_window_->getMap(), *frame_data, active_window_->getTracks(), output);
    }
  }
}

}  // namespace hydra::llm
