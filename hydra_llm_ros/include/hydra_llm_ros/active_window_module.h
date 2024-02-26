#pragma once
#include <config_utilities/factory.h>
#include <hydra/common/module.h>
#include <hydra/common/output_sink.h>
#include <hydra/reconstruction/reconstruction_output.h>
#include <hydra/utils/log_utilities.h>
#include <khronos/active_window/active_window.h>
#include <khronos_ros/visualization/active_window_visualizer.h>

namespace hydra::llm {

class ActiveWindowModule : public Module {
 public:
  using DataInputQueue = InputQueue<khronos::InputData::Ptr>;
  using OutputQueue = InputQueue<ReconstructionOutput::Ptr>;
  using Sink = OutputSink<uint64_t,
                          const Eigen::Isometry3d&,
                          const voxblox::Layer<voxblox::TsdfVoxel>&,
                          const ReconstructionOutput&>;

  struct Config {
    khronos::ActiveWindow::Config active_window;
    size_t max_queue_size = 0;
    bool use_visualizer = false;
    std::string active_window_visualizer_ns = "~reconstruction/active_window_viz";
    std::vector<Sink::Factory> sinks;
  };

  ActiveWindowModule(const Config& config, const OutputQueue::Ptr& output_queue);

  virtual ~ActiveWindowModule();

  void start() override;

  void stop() override;

  void save(const LogSetup& log_setup) override;

  std::string printInfo() const override;

  void spin();

  DataInputQueue::Ptr getInputQueue() const { return queue_; }

 public:
  const Config config;

 protected:
  std::unique_ptr<khronos::ActiveWindow> active_window_;

  std::atomic<bool> should_shutdown_ = false;
  DataInputQueue::Ptr queue_;
  OutputQueue::Ptr output_queue_;
  std::unique_ptr<std::thread> spin_thread_;
  std::unique_ptr<khronos::ActiveWindowVisualizer> visualizer_;
  Sink::List sinks_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<ActiveWindowModule,
                                     ActiveWindowModule,
                                     Config,
                                     OutputQueue::Ptr>("ActiveWindowModule");
};

void declare_config(ActiveWindowModule::Config& config);

}  // namespace hydra::llm
