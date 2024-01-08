#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/frontend/frontend_module.h>
#include <hydra_llm/places_clustering.h>
#include <llm/ClipVectorStamped.h>

namespace hydra::llm {

struct LLMFrontendConfig : public FrontendConfig {
  config::VirtualConfig<PlaceClustering> clustering;
};

void declare_config(LLMFrontendConfig& config);

class LLMFrontend : public FrontendModule {
 public:
  LLMFrontend(const LLMFrontendConfig& config,
              const SharedDsgInfo::Ptr& dsg,
              const SharedModuleState::Ptr& state,
              const LogSetup::Ptr& logs = nullptr);

  virtual ~LLMFrontend();

  const LLMFrontendConfig config;

 protected:
  void handleClipFeatures(const ::llm::ClipVectorStamped& msg);

  void updateActiveWindowViews(uint64_t curr_timestamp_ns);

  void updateImpl(const ReconstructionOutput& msg) override;

 protected:
  ros::NodeHandle nh_;
  ros::Subscriber clip_sub_;

  std::mutex clip_mutex_;
  std::list<ClipViewEmbedding::Ptr> keyframe_clip_vectors_;
  std::map<size_t, ClipView::Ptr> active_window_views_;
  PlaceClustering::Ptr places_clustering_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<FrontendModule,
                                     LLMFrontend,
                                     LLMFrontendConfig,
                                     SharedDsgInfo::Ptr,
                                     SharedModuleState::Ptr,
                                     LogSetup::Ptr>("LLMFrontend");
};

}  // namespace hydra::llm
