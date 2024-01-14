#pragma once
#include <config_utilities/factory.h>
#include <khronos/active_window/active_window.h>

#include "hydra/reconstruction/reconstruction_module.h"
#include "hydra/utils/log_utilities.h"

namespace hydra::llm {

class ActiveWindowModule : public ReconstructionModule {
 public:
  using OutputQueue = ReconstructionModule::OutputQueue;

  struct Config : public ReconstructionConfig {};

  ActiveWindowModule(const Config& config, const OutputQueue::Ptr& output_queue);

  virtual ~ActiveWindowModule();

  void start() override;

  void stop() override;

  void save(const LogSetup& log_setup) override;

  std::string printInfo() const override;

 public:
  const Config config;

 protected:
  inline static const auto registration_ =
      config::RegistrationWithConfig<ReconstructionModule,
                                     ActiveWindowModule,
                                     Config,
                                     OutputQueue::Ptr>("ActiveWindowModule");
};

void declare_config(ActiveWindowModule::Config& config);

}  // namespace hydra::llm
