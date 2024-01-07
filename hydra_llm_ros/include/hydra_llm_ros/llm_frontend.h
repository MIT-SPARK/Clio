#pragma once
#include <hydra/frontend/frontend_module.h>
#include <llm/ClipVectorStamped.h>

namespace hydra::llm {

struct ClipView {
  using Ptr = std::unique_ptr<ClipView>;
  Eigen::VectorXd embedding;
  uint64_t timestamp_ns;
};

struct LLMFrontendConfig : public FrontendConfig {};

class LLMFrontend : public FrontendModule {
 public:
  LLMFrontend(const LLMFrontendConfig& config,
              const SharedDsgInfo::Ptr& dsg,
              const SharedModuleState::Ptr& state,
              const LogSetup::Ptr& logs = nullptr);

  virtual ~LLMFrontend();

  const LLMFrontendConfig config;

 protected:
  virtual void initCallbacks() override;

  void updatePlaces(const ReconstructionOutput& msg);

  void handleClipFeatures(const ::llm::ClipVectorStamped& msg);

 protected:
  ros::Subscriber clip_sub_;
  std::mutex clip_mutex_;
  std::list<ClipView::Ptr> keyframe_clip_vectors_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<FrontendModule,
                                     LLMFrontend,
                                     LLMFrontendConfig,
                                     SharedDsgInfo::Ptr,
                                     SharedModuleState::Ptr,
                                     LogSetup::Ptr>("FrontendModule");
};

}  // namespace hydra::llm
