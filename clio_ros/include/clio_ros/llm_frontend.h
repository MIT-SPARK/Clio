#pragma once
#include <clio/scene_graph_types.h>
#include <config_utilities/virtual_config.h>
#include <hydra/frontend/frontend_module.h>

namespace clio {

struct LLMFrontendConfig : public hydra::FrontendModule::Config {
  double spatial_window_radius_m = 8.0;
  bool override_active_window = false;
};

void declare_config(LLMFrontendConfig& config);

class LLMFrontend : public hydra::FrontendModule {
 public:
  LLMFrontend(const LLMFrontendConfig& config,
              const hydra::SharedDsgInfo::Ptr& dsg,
              const hydra::SharedModuleState::Ptr& state,
              const hydra::LogSetup::Ptr& logs = nullptr);

  virtual ~LLMFrontend();

  const LLMFrontendConfig config;

  std::string printInfo() const override;

 protected:
  void initCallbacks() override;

  void updateImpl(const hydra::ReconstructionOutput::Ptr& msg) override;

  void updateKhronosObjects(const hydra::ReconstructionOutput& base_msg);

  void archiveObjects();

  void connectNewObjects();

 protected:
  std::set<NodeId> new_objects_;
  size_t object_id_ = 0;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<FrontendModule,
                                     LLMFrontend,
                                     LLMFrontendConfig,
                                     hydra::SharedDsgInfo::Ptr,
                                     hydra::SharedModuleState::Ptr,
                                     hydra::LogSetup::Ptr>("LLMFrontend");
};

}  // namespace clio
